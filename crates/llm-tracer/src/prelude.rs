use crate::ops::{linear_no_bias, repeat_kv};
use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};

/// Mode 0 (`--mode prelude`): Interactive sandbox for the 5 Linear Algebra & Tensor rules in the Prelude article
pub fn run_prelude_sandbox(dev: &Device) -> Result<()> {
    println!("=== Prelude Sandbox: 5 Linear Algebra & Tensor Rules for LLM Inference ===");

    // -------------------------------------------------------------------------
    // [1/5] From 1D Vector to 3D Hidden State [B, T, d] & Embedding Row Lookup
    // -------------------------------------------------------------------------
    println!("\n--- [1/5] Tensor Dimensions & Embedding Row Lookup (index_select) ---");
    // Imagine a mini vocabulary of V = 5 words, each represented by d = 4 floats: shape [5, 4]
    let embed_table = Tensor::from_slice(
        &[
            0.1f32, 0.2, 0.3, 0.4, // Row 0 (Token ID 0)
            1.0, 0.0, 1.0, 0.0, // Row 1 (Token ID 1)
            0.0, 2.0, 0.0, 1.0, // Row 2 (Token ID 2)
            1.0, 1.0, 0.0, 2.0, // Row 3 (Token ID 3)
            0.5, 0.5, 0.5, 0.5, // Row 4 (Token ID 4)
        ],
        (5, 4),
        dev,
    )?;
    // Input sentence with B = 1 batch, T = 3 tokens: [1, 2, 3]
    let input_ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), dev)?;
    let flat_ids = input_ids.flatten_all()?; // [1, 3] -> [3]
    let x = embed_table.index_select(&flat_ids, 0)?.reshape((1, 3, 4))?;
    println!("  embed_table shape: {:?} (Static Weight [V=5, d=4])", embed_table.dims());
    println!("  input_ids   shape: {:?} (Token IDs [B=1, T=3])", input_ids.dims());
    println!("  x (looked up)    : {:?} -> {:?}", x.dims(), x.i(0)?.to_vec2::<f32>()?);

    // -------------------------------------------------------------------------
    // [2/5] Vector Dot Product: Measuring Similarity Between Query and Keys
    // -------------------------------------------------------------------------
    println!("\n--- [2/5] Dot Product: Measuring Similarity (q · k) ---");
    // Query vector q = [2, 0], compared against 3 Key vectors: k1=[1, 1], k2=[2, 0], k3=[0, 2]
    let q = Tensor::from_slice(&[2.0f32, 0.0], (1, 2), dev)?;
    let keys = Tensor::from_slice(
        &[
            1.0f32, 1.0, // k1: 45-degree angle with q -> dot = 2*1 + 0*1 = 2.0
            2.0, 0.0, // k2: identical direction to q -> dot = 2*2 + 0*0 = 4.0 (highest!)
            0.0, 2.0, // k3: 90-degree orthogonal to q -> dot = 2*0 + 0*2 = 0.0
        ],
        (3, 2),
        dev,
    )?;
    let similarity = q.matmul(&keys.t()?)?; // [1, 2] x [2, 3] -> [1, 3]
    println!(
        "  q [1, 2] · keys^T [2, 3] -> scores {:?} = {:?}",
        similarity.dims(),
        similarity.to_vec2::<f32>()?
    );

    // -------------------------------------------------------------------------
    // [3/5] Matrix Multiplication & Middle-Dimension Cancellation ([M, K] x [K, N] -> [M, N])
    // -------------------------------------------------------------------------
    println!("\n--- [3/5] Linear Projection: [B, T, d_in] x [d_out, d_in]^T -> [B, T, d_out] ---");
    // Weight stored in SafeTensors as [d_out = 2, d_in = 4]
    let w_proj = Tensor::from_slice(
        &[
            1.0f32, 0.0, 1.0, 0.0, // Output feature 0 weights
            0.0, 1.0, 0.0, 1.0, // Output feature 1 weights
        ],
        (2, 4),
        dev,
    )?;
    let projected = linear_no_bias(&x, &w_proj)?; // [1, 3, 4] -> [1, 3, 2]
    println!(
        "  x {:?} projected by w_proj {:?} (via w_proj.t() [4, 2]) -> {:?}: {:?}",
        x.dims(),
        w_proj.dims(),
        projected.dims(),
        projected.i(0)?.to_vec2::<f32>()?
    );

    // -------------------------------------------------------------------------
    // [4/5] Matrix Multiplication (matmul) vs. Element-Wise Hadamard Product (*)
    // -------------------------------------------------------------------------
    println!("\n--- [4/5] MatMul vs. Element-Wise Multiplication (&gate * &up) ---");
    let gate = Tensor::from_slice(&[0.5f32, 2.0, 0.0, 1.5], (2, 2), dev)?;
    let up = Tensor::from_slice(&[4.0f32, 3.0, 9.0, 2.0], (2, 2), dev)?;
    let elem_wise = (&gate * &up)?; // Hadamard product: same shape [2, 2], position-by-position
    let mat_mul = gate.matmul(&up)?; // Matrix product: row-by-column dot products
    println!("  Element-wise (&gate * &up)  [2, 2]: {:?}", elem_wise.to_vec2::<f32>()?);
    println!("  Matrix mult (gate.matmul)   [2, 2]: {:?}", mat_mul.to_vec2::<f32>()?);

    // -------------------------------------------------------------------------
    // [5/5] Reshape, Transpose, Stride Contiguity (.contiguous()), and GQA Repeat
    // -------------------------------------------------------------------------
    println!("\n--- [5/5] Split Heads (reshape + transpose + contiguous) & GQA Broadcast ---");
    // Split x [B=1, T=3, d=4] into h=2 heads of d_h=2: [1, 3, 4] -> [1, 3, 2, 2] -> [1, 2, 3, 2]
    let reshaped = x.reshape((1, 3, 2, 2))?;
    let transposed = reshaped.transpose(1, 2)?;
    println!(
        "  After reshape [1, 3, 2, 2]: is_contiguous = {} | strides = {:?}",
        reshaped.is_contiguous(),
        reshaped.stride()
    );
    println!(
        "  After transpose(1, 2) [1, 2, 3, 2]: is_contiguous = {} | strides = {:?}",
        transposed.is_contiguous(),
        transposed.stride()
    );
    let contiguous = transposed.contiguous()?;
    println!(
        "  After .contiguous()   [1, 2, 3, 2]: is_contiguous = {} | strides = {:?}",
        contiguous.is_contiguous(),
        contiguous.stride()
    );

    // GQA repeat_kv: expand 1 KV head [1, 1, 3, 2] to 3 Query heads [1, 3, 3, 2]
    let single_kv = contiguous.narrow(1, 0, 1)?; // Take head 0 -> [1, 1, 3, 2]
    let repeated_kv = repeat_kv(&single_kv, 3)?; // Repeat 3x -> [1, 3, 3, 2]
    println!(
        "  GQA repeat_kv: {:?} repeated 3x -> {:?}",
        single_kv.dims(),
        repeated_kv.dims()
    );

    Ok(())
}
