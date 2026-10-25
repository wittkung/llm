use candle_core::{DType, Device, Result, Tensor};

/// Marks whether a tensor in memory is a static model parameter, a dynamic activation, or a KV cache slice.
#[derive(Clone, Copy)]
pub enum Role {
    Weight,     // Static parameter tensor from `.safetensors`: shape has only (V, d, h, d_h, d_ff), never B or T
    Activation, // Dynamic intermediate tensor: shape always carries batch B and sequence length T
    KvCache,    // Historical K/V activation tensor cached across autoregressive steps
}

/// Prints a single white-box trace line: `[Stage] Name | Role | Shape | [first 4 floats...]`
pub fn trace(stage: &str, name: &str, role: Role, t: &Tensor) -> Result<()> {
    let role_tag = match role {
        Role::Weight => "WEIGHT    ",
        Role::Activation => "ACTIVATION",
        Role::KvCache => "KV_CACHE  ",
    };
    let elem_count = t.elem_count().min(4);
    let flat_slice = t
        .flatten_all()?
        .narrow(0, 0, elem_count)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let head: Vec<String> = flat_slice.iter().map(|v| format!("{:+.4}", v)).collect();
    println!(
        "[{:<14}] {:<36} | {} | shape: {:<20} | [{} ...]",
        stage,
        name,
        role_tag,
        format!("{:?}", t.dims()),
        head.join(", ")
    );
    Ok(())
}

/// Operator 1 — Linear Projection without Bias:
/// Flattens 3D activation `[B, T, d_in]` to 2D `[B * T, d_in]`, multiplies by transposed weight
/// `weight.t()?` of shape `[d_in, d_out]`, and reshapes back to `[B, T, d_out]`.
pub fn linear_no_bias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (b, t, d_in) = x.dims3()?;
    let (d_out, _) = weight.dims2()?;
    x.reshape((b * t, d_in))?
        .matmul(&weight.t()?)?
        .reshape((b, t, d_out))
}

/// Operator 2 — RMSNorm (Root Mean Square Layer Normalization):
/// Computes `x / sqrt(mean(x^2) + eps) * weight`, preserving shape `[B, T, d]`.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    candle_nn::ops::rms_norm(x, weight, eps as f32)
}

/// Operator 3 — RoPE Frequency Table Precomputation:
/// Builds `cos` and `sin` tables of shape `[max_seq_len, d_h / 2]` where
/// `inv_freq[i] = 1.0 / theta^(2i / d_h)` and `freqs[t, i] = t * inv_freq[i]`.
pub fn precompute_rope_tables(
    max_seq_len: usize,
    d_h: usize,
    theta: f32,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = d_h / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / d_h as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
    let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
    let t = Tensor::from_vec(positions, (max_seq_len, 1), dev)?;
    let freqs = t.matmul(&inv_freq)?; // Outer product: [max_seq_len, 1] x [1, 32] -> [max_seq_len, 32]
    Ok((freqs.cos()?, freqs.sin()?))
}

/// Operator 4 — GQA Key/Value Head Expansion (`repeat_kv`):
/// Expands `[B, h_kv, T, d_h]` by repeating each KV head `n_rep = h_q / h_kv` times -> `[B, h_q, T, d_h]`.
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, h_kv, seq_len, d_h) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h_kv, n_rep, seq_len, d_h))?
        .reshape((b, h_kv * n_rep, seq_len, d_h))
}

/// Operator 5 — Upper-Triangular Causal Mask:
/// Sets all future positions `j > i + offset` to `-inf` so `exp(-inf) = 0` in Softmax.
/// During single-token Decode (`q_len == 1`), no future positions exist, so returns `scores` as-is.
pub fn apply_causal_mask(scores: &Tensor, offset: usize) -> Result<Tensor> {
    let (b, h, q_len, kv_len) = scores.dims4()?;
    if q_len == 1 {
        return Ok(scores.clone());
    }
    let mut mask_vec = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            if j > i + offset {
                mask_vec[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(mask_vec, (1, 1, q_len, kv_len), scores.device())?
        .broadcast_as((b, h, q_len, kv_len))?;
    scores.broadcast_add(&mask)
}
