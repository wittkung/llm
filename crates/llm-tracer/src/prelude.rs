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
    // [3/6] Matrix Multiplication ([M, K] x [K, N] -> [M, N]) & 3D-to-2D Flatten
    // -------------------------------------------------------------------------
    println!("\n--- [3/6] Matrix Multiplication, Transpose & 3D <-> 2D Flatten/Unflatten ---");
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
        "  2D MatMul: picked_2d {:?} x w^T {:?} -> out_2d {:?}: {:?}",
        picked_2d.dims(),
        w_t.dims(),
        out_2d.dims(),
        out_2d.to_vec2::<f32>()?
    );

    // 3D [B=2, M=2, K=3] multiplied by 2D [K=3, N=2]:
    // 2 classes (B=2), 2 students per class (M=2), 3 scores per student (K=3)
    let x_3d = Tensor::from_slice(
        &[
            // Class 0 (2 students, 3 scores each)
            1.0f32, 2.0, 3.0, // Class 0 - Student A
            4.0, 5.0, 6.0, // Class 0 - Student B
            // Class 1 (2 students, 3 scores each)
            10.0, 20.0, 30.0, // Class 1 - Student C
            40.0, 50.0, 60.0, // Class 1 - Student D
        ],
        (2, 2, 3),
        dev,
    )?;
    let weight_2d = Tensor::from_slice(
        &[
            1.0f32, 1.0, // col 0 = sum of 3 scores; col 1 = first score only
            1.0, 0.0, //
            1.0, 0.0, //
        ],
        (3, 2),
        dev,
    )?;
    // Step 1 (3D -> 2D): Merge B=2 and M=2 into B*M=4 rows -> [4, 3]
    let x_flat_2d = x_3d.reshape((2 * 2, 3))?;
    // Step 2 (2D MatMul): [4, 3] x [3, 2] -> [4, 2]
    let y_flat_2d = x_flat_2d.matmul(&weight_2d)?;
    // Step 3 (2D -> 3D): Split 4 rows back into B=2 classes of M=2 students -> [2, 2, 2]
    let y_3d = y_flat_2d.reshape((2, 2, 2))?;
    println!("  3D -> 2D Flatten   x_flat_2d {:?}: {:?}", x_flat_2d.dims(), x_flat_2d.to_vec2::<f32>()?);
    println!("  2D MatMul          y_flat_2d {:?}: {:?}", y_flat_2d.dims(), y_flat_2d.to_vec2::<f32>()?);
    println!("  2D -> 3D Unflatten y_3d      {:?}: {:?}", y_3d.dims(), y_3d.to_vec3::<f32>()?);

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
    // [5/6] Reshape (Grouping), Transpose (Swapping Hierarchy), & Contiguous
    // -------------------------------------------------------------------------
    println!("\n--- [5/6] Reshape, Transpose & Contiguous (3 Students x 4 Courses) ---");
    // 3 students, 4 courses each (tens digit = student 1..3, ones digit = course 1..4)
    // Courses 1,2 = Arts group; Courses 3,4 = Science group
    let scores_2d = Tensor::from_slice(
        &[
            11.0f32, 12.0, 13.0, 14.0, // Student 1
            21.0, 22.0, 23.0, 24.0, // Student 2
            31.0, 32.0, 33.0, 34.0, // Student 3
        ],
        (3, 4),
        dev,
    )?;
    println!("  1. Original [3, 4] (3 students, 4 courses):");
    println!("     {:?}", scores_2d.to_vec2::<f32>()?);

    // Step A: reshape [3, 4] -> [3, 2, 2] (3 students, each split into 2 groups of 2 courses)
    let reshaped = scores_2d.reshape((3, 2, 2))?;
    println!(
        "  2. After reshape(3, 2, 2) [3 students, 2 groups, 2 courses] (is_contiguous = {}):",
        reshaped.is_contiguous()
    );
    println!("     {:?}", reshaped.to_vec3::<f32>()?);

    // Step B: transpose(0, 1) -> [2, 3, 2] (Swap outer hierarchy to: 2 groups, 3 students, 2 courses)
    let transposed = reshaped.transpose(0, 1)?;
    println!(
        "  3. After transpose(0, 1)  [2 groups, 3 students, 2 courses] (is_contiguous = {}):",
        transposed.is_contiguous()
    );
    println!("     Group 0 (Arts    for all 3 students): {:?}", transposed.i(0)?.to_vec2::<f32>()?);
    println!("     Group 1 (Science for all 3 students): {:?}", transposed.i(1)?.to_vec2::<f32>()?);

    // Step C: .contiguous() physically reorders memory to match the new [2, 3, 2] grouping
    let contiguous = transposed.contiguous()?;
    println!(
        "  4. After .contiguous()    [2, 3, 2] (is_contiguous = {}):",
        contiguous.is_contiguous()
    );
    println!("     Flattened physical order: {:?}", contiguous.flatten_all()?.to_vec1::<f32>()?);

    // Step D: Broadcast expand (repeat each of the 2 groups 3 times -> 6 groups)
    let repeated = contiguous
        .unsqueeze(1)?
        .expand((2, 3, 3, 2))?
        .reshape((6, 3, 2))?;
    println!(
        "  5. Broadcast expand: {:?} (2 groups) each repeated 3x -> {:?} (6 groups)",
        contiguous.dims(),
        repeated.dims()
    );

    // -------------------------------------------------------------------------
    // [6/6] RMS Normalization & Softmax (with -inf masking)
    // -------------------------------------------------------------------------
    println!("\n--- [6/6] RMS Normalization & Softmax ---");
    let scale = Tensor::ones((4,), candle_core::DType::F32, dev)?;
    let v_normal = Tensor::from_slice(&[2.0f32, -4.0, 4.0, 8.0], (1, 1, 4), dev)?;
    let v_huge = Tensor::from_slice(&[200.0f32, -400.0, 400.0, 800.0], (1, 1, 4), dev)?;
    let rms_normal = candle_nn::ops::rms_norm(&v_normal, &scale, 1e-5)?;
    let rms_huge = candle_nn::ops::rms_norm(&v_huge, &scale, 1e-5)?;
    println!(
        "  v_normal [2, -4, 4, 8]         (RMS = 5)   -> RMSNorm: {:?}",
        rms_normal.flatten_all()?.to_vec1::<f32>()?
    );
    println!(
        "  v_huge   [200, -400, 400, 800] (RMS = 500) -> RMSNorm: {:?}",
        rms_huge.flatten_all()?.to_vec1::<f32>()?
    );

    let scores_a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 3), dev)?;
    let probs_a = candle_nn::ops::softmax_last_dim(&scores_a)?;
    println!(
        "  Scores [1.0, 2.0, 3.0]  -> Softmax: {:?}",
        probs_a.flatten_all()?.to_vec1::<f32>()?
    );

    let scores_b = Tensor::from_slice(&[1.0f32, 1.0, f32::NEG_INFINITY], (1, 3), dev)?;
    let probs_b = candle_nn::ops::softmax_last_dim(&scores_b)?;
    println!(
        "  Scores [1.0, 1.0, -inf] -> Softmax: {:?}",
        probs_b.flatten_all()?.to_vec1::<f32>()?
    );

    Ok(())
}
