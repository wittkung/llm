use crate::inspect::{get_tensor, load_config_and_weights, LlamaConfig};
use crate::ops::{
    apply_causal_mask, linear_no_bias, precompute_rope_tables, repeat_kv, rms_norm, trace, Role,
};
use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Executes one complete forward pass through all 30 layers of `SmolLM2-135M-Instruct` (Stages 01 — 10).
/// - In Step 1 (Prefill): `input_ids` has shape `[1, 10]`, `offset = 0`.
/// - In Step 2 (Decode):  `input_ids` has shape `[1, 1]`,  `offset = 10`.
#[allow(clippy::too_many_arguments)]
pub fn forward_step(
    step_label: &str,
    input_ids: &Tensor,
    offset: usize,
    cfg: &LlamaConfig,
    weights: &HashMap<String, Tensor>,
    cos_all: &Tensor,
    sin_all: &Tensor,
    kv_caches: &mut [Option<(Tensor, Tensor)>],
) -> Result<u32> {
    println!("\n==========================================================================================");
    println!("=== {} (offset = {}) ===", step_label, offset);
    println!("==========================================================================================");

    let (b, seq_len) = input_ids.dims2()?;
    let d = cfg.hidden_size;
    let h_q = cfg.num_attention_heads;
    let h_kv = cfg.num_key_value_heads;
    let d_h = cfg.head_dim();
    let n_rep = h_q / h_kv;

    // Stage 01: Input Token IDs [B, T]
    trace("01.Tokens", "input_ids", Role::Activation, input_ids)?;

    // Stage 02: Embedding Table Lookup ([B, T] -> [B, T, 576])
    let w_embed = get_tensor(weights, "model.embed_tokens.weight")?;
    trace("02.Embedding", "model.embed_tokens.weight", Role::Weight, w_embed)?;
    let flat_ids = input_ids.flatten_all()?;
    let mut hidden = w_embed.index_select(&flat_ids, 0)?.reshape((b, seq_len, d))?;
    trace("02.Embedding", "X_0 (embedded hidden state)", Role::Activation, &hidden)?;

    // Slice precomputed RoPE cos/sin tables for current position window [offset .. offset + seq_len]
    let cos = cos_all.narrow(0, offset, seq_len)?;
    let sin = sin_all.narrow(0, offset, seq_len)?;

    // Stage 03 — 08: Loop through all 30 Transformer Layers (Layer 0 is traced in full detail)
    for (l, kv_slot) in kv_caches.iter_mut().enumerate().take(cfg.num_hidden_layers) {
        let p = format!("model.layers.{l}");
        let trace_layer = l == 0;

        // 1. Pre-Attention RMSNorm
        let w_ln1 = get_tensor(weights, &format!("{p}.input_layernorm.weight"))?;
        let x_norm = rms_norm(&hidden, w_ln1, cfg.rms_norm_eps)?;
        if trace_layer {
            trace("03.PreRMSNorm", &format!("{p}.input_layernorm.weight"), Role::Weight, w_ln1)?;
            trace("03.PreRMSNorm", "X_norm (Layer 0)", Role::Activation, &x_norm)?;
        }

        // 2. Stage 04: Q, K, V Linear Projections & Split Heads
        let w_q = get_tensor(weights, &format!("{p}.self_attn.q_proj.weight"))?;
        let w_k = get_tensor(weights, &format!("{p}.self_attn.k_proj.weight"))?;
        let w_v = get_tensor(weights, &format!("{p}.self_attn.v_proj.weight"))?;
        let q_lin = linear_no_bias(&x_norm, w_q)?;
        let k_lin = linear_no_bias(&x_norm, w_k)?;
        let v_lin = linear_no_bias(&x_norm, w_v)?;

        if trace_layer {
            trace("04.AttnProj", &format!("{p}.self_attn.q_proj.weight"), Role::Weight, w_q)?;
            trace("04.AttnProj", &format!("{p}.self_attn.k_proj.weight"), Role::Weight, w_k)?;
            trace("04.AttnProj", &format!("{p}.self_attn.v_proj.weight"), Role::Weight, w_v)?;
            trace("04.AttnProj", "Q_lin", Role::Activation, &q_lin)?;
            trace("04.AttnProj", "K_lin", Role::Activation, &k_lin)?;
            trace("04.AttnProj", "V_lin", Role::Activation, &v_lin)?;
        }

        let q = q_lin.reshape((b, seq_len, h_q, d_h))?.transpose(1, 2)?.contiguous()?;
        let k = k_lin.reshape((b, seq_len, h_kv, d_h))?.transpose(1, 2)?.contiguous()?;
        let v = v_lin.reshape((b, seq_len, h_kv, d_h))?.transpose(1, 2)?.contiguous()?;

        // 3. Stage 03/04: Apply RoPE Rotary Position Embedding to Q and K
        let q_rope = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_rope = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
        if trace_layer {
            trace("03.RoPE", "Q_rope (rotated queries)", Role::Activation, &q_rope)?;
            trace("03.RoPE", "K_rope (rotated keys)", Role::Activation, &k_rope)?;
        }

        // 4. KV Cache: Concatenate new K/V along sequence dimension (dim = 2)
        let (k_cache, v_cache) = match kv_slot.take() {
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[&prev_k, &k_rope], 2)?,
                Tensor::cat(&[&prev_v, &v], 2)?,
            ),
            None => (k_rope, v),
        };
        if trace_layer {
            trace("05.KVCache", "Layer 0 K_cache", Role::KvCache, &k_cache)?;
            trace("05.KVCache", "Layer 0 V_cache", Role::KvCache, &v_cache)?;
        }
        *kv_slot = Some((k_cache.clone(), v_cache.clone()));

        // 5. Stage 05: GQA Repeat KV, Scaled Dot-Product, Causal Mask, Softmax, and Context
        let k_full = repeat_kv(&k_cache, n_rep)?;
        let v_full = repeat_kv(&v_cache, n_rep)?;
        let scale = 1.0f64 / (d_h as f64).sqrt();
        let attn_scores = (q_rope.matmul(&k_full.t()?)? * scale)?;
        let masked_scores = apply_causal_mask(&attn_scores, offset)?;
        let attn_probs = candle_nn::ops::softmax_last_dim(&masked_scores)?;
        let context = attn_probs.matmul(&v_full.contiguous()?)?;
        if trace_layer {
            trace("05.Attention", "K_full (after repeat_kv)", Role::Activation, &k_full)?;
            trace("05.Attention", "Scores (QK^T / sqrt(d_h))", Role::Activation, &masked_scores)?;
            trace("05.Attention", "Probs (Softmax)", Role::Activation, &attn_probs)?;
            trace("05.Attention", "Context (Probs * V)", Role::Activation, &context)?;
        }

        // 6. Stage 06: Merge Heads, Output Projection W_O, and First Residual Addition
        let o_concat = context.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, d))?;
        let w_o = get_tensor(weights, &format!("{p}.self_attn.o_proj.weight"))?;
        let attn_out = linear_no_bias(&o_concat, w_o)?;
        let h_mid = (&hidden + &attn_out)?;
        if trace_layer {
            trace("06.MergeOut", "O_concat", Role::Activation, &o_concat)?;
            trace("06.MergeOut", &format!("{p}.self_attn.o_proj.weight"), Role::Weight, w_o)?;
            trace("06.MergeOut", "H_mid (after Attn residual)", Role::Activation, &h_mid)?;
        }

        // 7. Stage 07: Post-Attention RMSNorm, SwiGLU MLP, and Second Residual Addition
        let w_ln2 = get_tensor(weights, &format!("{p}.post_attention_layernorm.weight"))?;
        let h_norm = rms_norm(&h_mid, w_ln2, cfg.rms_norm_eps)?;
        let w_gate = get_tensor(weights, &format!("{p}.mlp.gate_proj.weight"))?;
        let w_up = get_tensor(weights, &format!("{p}.mlp.up_proj.weight"))?;
        let w_down = get_tensor(weights, &format!("{p}.mlp.down_proj.weight"))?;

        let gate = candle_nn::ops::silu(&linear_no_bias(&h_norm, w_gate)?)?;
        let up = linear_no_bias(&h_norm, w_up)?;
        let gated = (&gate * &up)?;
        let mlp_out = linear_no_bias(&gated, w_down)?;
        hidden = (&h_mid + &mlp_out)?;

        if trace_layer {
            trace("07.SwiGLU_MLP", &format!("{p}.mlp.gate_proj.weight"), Role::Weight, w_gate)?;
            trace("07.SwiGLU_MLP", "Gate = SiLU(H_norm * W_gate)", Role::Activation, &gate)?;
            trace("07.SwiGLU_MLP", "Up = H_norm * W_up", Role::Activation, &up)?;
            trace("07.SwiGLU_MLP", &format!("{p}.mlp.down_proj.weight"), Role::Weight, w_down)?;
            trace("07.SwiGLU_MLP", "X_1 (Layer 0 output)", Role::Activation, &hidden)?;
        }
    }

    // Stage 08: Final RMSNorm after exiting Layer 29
    let w_final_norm = get_tensor(weights, "model.norm.weight")?;
    let h_final = rms_norm(&hidden, w_final_norm, cfg.rms_norm_eps)?;
    trace("08.RepeatL30", "X_30 (after 30 layers + norm)", Role::Activation, &h_final)?;

    // Stage 09: Output Projection (LM Head tied to model.embed_tokens.weight)
    let w_lm_head = weights.get("lm_head.weight").unwrap_or(w_embed);
    trace("09.LM_Head", "lm_head.weight (tied W_E)", Role::Weight, w_lm_head)?;
    let logits = linear_no_bias(&h_final, w_lm_head)?;
    trace("09.LM_Head", "Logits [B, T, V]", Role::Activation, &logits)?;

    // Stage 10: Slice last token logits [V] and pick greedy argmax token ID
    let last_logits = logits.i((0, seq_len - 1))?;
    trace("10.SelectNext", "last_logits [V]", Role::Activation, &last_logits)?;
    let next_token_id = last_logits.argmax(0)?.to_scalar::<u32>()?;
    println!("[10.SelectNext ] Greedy argmax next_token_id = {}", next_token_id);

    Ok(next_token_id)
}

/// Mode 3 (`--mode real`): Tokenizes the ChatML prompt and runs Step 1 (Prefill) + Step 2 (Decode)
pub fn run_real_trace(model_dir: &Path, dev: &Device) -> Result<()> {
    let (cfg, weights) = load_config_and_weights(model_dir, dev)?;
    let (cos_all, sin_all) = precompute_rope_tables(128, cfg.head_dim(), cfg.rope_theta, dev)?;
    let mut kv_caches: Vec<Option<(Tensor, Tensor)>> = vec![None; cfg.num_hidden_layers];

    // Load `tokenizer.json` and encode the ChatML prompt into 10 Token IDs
    let tok_path = model_dir.join("tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("Failed to load {}: {e}", tok_path.display()))?;

    let raw_prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    let enc = tok
        .encode(raw_prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {e}"))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    println!(
        "\n[01.Tokenizer ] Raw prompt: {:?}\n[01.Tokenizer ] Tokens ({}): {:?}\n[01.Tokenizer ] Token IDs : {:?}",
        raw_prompt,
        prompt_ids.len(),
        enc.get_tokens(),
        prompt_ids
    );

    // Step 1: Prefill all 10 prompt tokens at once (shape [1, 10], offset = 0)
    let prefill_input = Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), dev)?;
    let first_generated_id = forward_step(
        &format!("STEP 1: PREFILL (T = {} ChatML prompt tokens)", prompt_ids.len()),
        &prefill_input,
        0,
        &cfg,
        &weights,
        &cos_all,
        &sin_all,
        &mut kv_caches,
    )?;
    let first_piece = tok
        .decode(&[first_generated_id], false)
        .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;
    println!(
        "[10.Decoded    ] Step 1 generated token {} -> {:?}",
        first_generated_id, first_piece
    );

    // Step 2: Autoregressive Decode feeding only the 1 new token (shape [1, 1], offset = 10)
    let decode_input = Tensor::from_slice(&[first_generated_id], (1, 1), dev)?;
    let second_generated_id = forward_step(
        &format!(
            "STEP 2: AUTOREGRESSIVE DECODE (T_new = 1, KV Cache {} -> {})",
            prompt_ids.len(),
            prompt_ids.len() + 1
        ),
        &decode_input,
        prompt_ids.len(),
        &cfg,
        &weights,
        &cos_all,
        &sin_all,
        &mut kv_caches,
    )?;
    let second_piece = tok
        .decode(&[second_generated_id], false)
        .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;
    let full_reply = tok
        .decode(&[first_generated_id, second_generated_id], false)
        .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;
    println!(
        "[10.Decoded    ] Step 2 generated token {} -> {:?}",
        second_generated_id, second_piece
    );
    println!(
        "[11.Summary    ] First 2 generated tokens decoded: {:?}",
        full_reply
    );

    Ok(())
}
