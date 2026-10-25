use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};

/// Mode 0 (`--mode prelude`): Pure Linear Algebra & Tensor programming sandbox (zero LLM jargon)
pub fn run_prelude_sandbox(dev: &Device) -> Result<()> {
    println!("=== Prelude Sandbox: Pure Linear Algebra & Tensor Mechanics in Rust ===");

    // -------------------------------------------------------------------------
    // [1/6] Multi-dimensional Arrays (Tensors) & Row Indexing (index_select)
    // -------------------------------------------------------------------------
    println!("\n--- [1/6] Tensor Shapes (1D / 2D / 3D) & Row Lookup (index_select) ---");
    // A 2D matrix `table` of shape [5, 4] (5 rows, 4 columns)
    let table = Tensor::from_slice(
        &[
            0.1f32, 0.2, 0.3, 0.4, // Row 0
            1.0, 0.0, 1.0, 0.0, // Row 1
            0.0, 2.0, 0.0, 1.0, // Row 2
            1.0, 1.0, 0.0, 2.0, // Row 3
            0.5, 0.5, 0.5, 0.5, // Row 4
        ],
        (5, 4),
        dev,
    )?;
    // Pick rows [1, 2, 3] and pack them into a 3D tensor of shape [1, 3, 4]
    let row_indices = Tensor::from_slice(&[1u32, 2, 3], (3,), dev)?;
    let picked_2d = table.index_select(&row_indices, 0)?; // [3, 4]
    let tensor_3d = picked_2d.reshape((1, 3, 4))?; // [1, 3, 4]
    println!("  2D table     shape: {:?}", table.dims());
    println!("  row_indices  shape: {:?}", row_indices.dims());
    println!(
        "  tensor_3d    shape: {:?} -> {:?}",
        tensor_3d.dims(),
        tensor_3d.i(0)?.to_vec2::<f32>()?
    );

    // -------------------------------------------------------------------------
    // [2/6] Vector Dot Product: Algebraic Sum & Geometric Cosine Similarity
    // -------------------------------------------------------------------------
    println!("\n--- [2/6] Vector Dot Product (a · b) ---");
    // Vector a = [2, 0], compared against b1 = [1, 1], b2 = [2, 0], b3 = [0, 2]
    let a = Tensor::from_slice(&[2.0f32, 0.0], (1, 2), dev)?;
    let b_list = Tensor::from_slice(
        &[
            1.0f32, 1.0, // b1 (45 deg): 2*1 + 0*1 = 2.0
            2.0, 0.0, // b2 (0 deg, same direction): 2*2 + 0*0 = 4.0
            0.0, 2.0, // b3 (90 deg, perpendicular): 2*0 + 0*2 = 0.0
        ],
        (3, 2),
        dev,
    )?;
    let dot_products = a.matmul(&b_list.t()?)?; // [1, 2] x [2, 3] -> [1, 3]
    println!(
        "  a [1, 2] · b_list^T [2, 3] -> {:?} = {:?}",
        dot_products.dims(),
        dot_products.to_vec2::<f32>()?
    );

    // -------------------------------------------------------------------------
    // [3/6] Matrix Multiplication ([M, K] x [K, N] -> [M, N]) & Transpose (.t())
    // -------------------------------------------------------------------------
    println!("\n--- [3/6] Matrix Multiplication & Middle-Dimension Cancellation ---");
    // w has shape [2, 4]; to multiply with picked_2d [3, 4], we transpose w to [4, 2]
    let w = Tensor::from_slice(
        &[
            1.0f32, 0.0, 1.0, 0.0, // Row 0
            0.0, 1.0, 0.0, 1.0, // Row 1
        ],
        (2, 4),
        dev,
    )?;
    let w_t = w.t()?; // [4, 2]
    let out_2d = picked_2d.matmul(&w_t)?; // [3, 4] x [4, 2] -> [3, 2]
    println!(
        "  picked_2d {:?} x w^T {:?} -> out_2d {:?}: {:?}",
        picked_2d.dims(),
        w_t.dims(),
        out_2d.dims(),
        out_2d.to_vec2::<f32>()?
    );

    // -------------------------------------------------------------------------
    // [4/6] Matrix Multiplication (matmul) vs. Element-Wise Product (&m1 * &m2)
    // -------------------------------------------------------------------------
    println!("\n--- [4/6] MatMul (m1.matmul(&m2)) vs. Element-Wise (&m1 * &m2) ---");
    let m1 = Tensor::from_slice(&[0.5f32, 2.0, 0.0, 1.5], (2, 2), dev)?;
    let m2 = Tensor::from_slice(&[4.0f32, 3.0, 9.0, 2.0], (2, 2), dev)?;
    let elem_wise = (&m1 * &m2)?; // Hadamard product
    let mat_mul = m1.matmul(&m2)?; // Matrix product
    println!("  Element-wise (&m1 * &m2)   [2, 2]: {:?}", elem_wise.to_vec2::<f32>()?);
    println!("  Matrix mult (m1.matmul)    [2, 2]: {:?}", mat_mul.to_vec2::<f32>()?);

    // -------------------------------------------------------------------------
    // [5/6] Memory Layout: reshape, transpose, stride,contiguous, and expand
    // -------------------------------------------------------------------------
    println!("\n--- [5/6] Memory Strides (reshape vs. transpose vs. contiguous) & Broadcast ---");
    let reshaped = tensor_3d.reshape((1, 3, 2, 2))?;
    let transposed = reshaped.transpose(1, 2)?;
    let contiguous = transposed.contiguous()?;
    println!(
        "  After reshape [1, 3, 2, 2]      : is_contiguous = {:<5} | strides = {:?}",
        reshaped.is_contiguous(),
        reshaped.stride()
    );
    println!(
        "  After transpose(1, 2) [1, 2, 3, 2]: is_contiguous = {:<5} | strides = {:?}",
        transposed.is_contiguous(),
        transposed.stride()
    );
    println!(
        "  After .contiguous()   [1, 2, 3, 2]: is_contiguous = {:<5} | strides = {:?}",
        contiguous.is_contiguous(),
        contiguous.stride()
    );

    let slice_one = contiguous.narrow(1, 0, 1)?; // [1, 1, 3, 2]
    let expanded = slice_one
        .unsqueeze(2)?
        .expand((1, 1, 3, 3, 2))?
        .reshape((1, 3, 3, 2))?;
    println!(
        "  Broadcast expand: {:?} repeated 3x along axis -> {:?}",
        slice_one.dims(),
        expanded.dims()
    );

    // -------------------------------------------------------------------------
    // [6/6] RMS Normalization & Softmax (with -inf masking)
    // -------------------------------------------------------------------------
    println!("\n--- [6/6] RMS Normalization & Softmax ---");
    let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 2.0], (1, 1, 4), dev)?;
    let scale = Tensor::ones((4,), candle_core::DType::F32, dev)?;
    let v_rms = candle_nn::ops::rms_norm(&v, &scale, 1e-5)?;
    println!(
        "  Input vector [1, 2, 3, 2] -> after RMSNorm: {:?}",
        v_rms.flatten_all()?.to_vec1::<f32>()?
    );

    let scores = Tensor::from_slice(&[1.0f32, 1.0, f32::NEG_INFINITY], (1, 3), dev)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    println!(
        "  Scores [1.0, 1.0, -inf]   -> after Softmax: {:?}",
        probs.flatten_all()?.to_vec1::<f32>()?
    );

    Ok(())
}
