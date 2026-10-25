use crate::ops::{apply_causal_mask, trace, Role};
use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};

/// Mode 2 (`--mode toy`): Verifies the exact hand-calculated Toy MHA matrices from Section 6
pub fn run_toy_verification(dev: &Device) -> Result<()> {
    println!("=== Episode 0 Toy Hand-Calculation Verification (B=1, T=3, d=4, h=2, d_h=2) ===");
    let (b, t, d, h, d_h) = (1, 3, 4, 2, 2);

    // Step 1: Input hidden state X of shape [1, 3, 4] for ["我", "喜欢", "写"]
    let x = Tensor::from_slice(
        &[
            1.0f32, 0.0, 1.0, 0.0, // t=0 ("我")
            0.0, 1.0, 1.0, 0.0,    // t=1 ("喜欢")
            1.0, 1.0, 0.0, 1.0,    // t=2 ("写")
        ],
        (b, t, d),
        dev,
    )?;
    trace("02.Embedding", "X (toy input states)", Role::Activation, &x)?;

    // Step 2: Parameter matrices W_Q, W_K, W_V of shape [4, 4] (in math layout [d_in, d_out])
    let w_q_math = Tensor::from_slice(
        &[
            1.0f32, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
        ],
        (d, d),
        dev,
    )?;
    let w_k_math = Tensor::from_slice(
        &[
            1.0f32, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 0.0, 1.0,
        ],
        (d, d),
        dev,
    )?;
    let w_v_math = Tensor::from_slice(
        &[
            1.0f32, 0.0, 2.0, 0.0,
            0.0, 1.0, 0.0, 2.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
        ],
        (d, d),
        dev,
    )?;

    let q_raw = x.reshape((b * t, d))?.matmul(&w_q_math)?.reshape((b, t, d))?;
    let k_raw = x.reshape((b * t, d))?.matmul(&w_k_math)?.reshape((b, t, d))?;
    let v_raw = x.reshape((b * t, d))?.matmul(&w_v_math)?.reshape((b, t, d))?;

    trace("04.AttnProj", "Q_raw", Role::Activation, &q_raw)?;
    trace("04.AttnProj", "K_raw", Role::Activation, &k_raw)?;
    trace("04.AttnProj", "V_raw", Role::Activation, &v_raw)?;

    // Split into h = 2 heads of width d_h = 2: [1, 3, 4] -> [1, 3, 2, 2] -> [1, 2, 3, 2]
    let q = q_raw.reshape((b, t, h, d_h))?.transpose(1, 2)?.contiguous()?;
    let k = k_raw.reshape((b, t, h, d_h))?.transpose(1, 2)?.contiguous()?;
    let v = v_raw.reshape((b, t, h, d_h))?.transpose(1, 2)?.contiguous()?;

    // Step 3 & 4: Scaled dot-product attention, causal mask, Softmax, and head concatenation
    let scale = 1.0f64 / (d_h as f64).sqrt();
    let scaled_scores = (&q.matmul(&k.t()?)? * scale)?;
    let masked_scores = apply_causal_mask(&scaled_scores, 0)?;
    let attn_probs = candle_nn::ops::softmax_last_dim(&masked_scores)?;
    let context = attn_probs.matmul(&v)?;
    let o_concat = context.transpose(1, 2)?.contiguous()?.reshape((b, t, d))?;

    println!("\n--- Head 1 Attention Probs A^(1) [3x3] ---");
    println!("{:?}", attn_probs.i((0, 0))?.to_vec2::<f32>()?);
    println!("--- Head 2 Attention Probs A^(2) [3x3] ---");
    println!("{:?}", attn_probs.i((0, 1))?.to_vec2::<f32>()?);
    println!("--- Concatenated Output O_concat [3x4] ---");
    println!("{:?}", o_concat.i(0)?.to_vec2::<f32>()?);

    Ok(())
}
