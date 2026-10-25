use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Mode {
    /// Inspect the 272 parameter tensors and exact byte/parameter budget of SmolLM2-135M-Instruct
    Inspect,
    /// Run the exact hand-calculated Toy MHA example (B=1, T=3, d=4, h=2, d_h=2) from Section 5
    Toy,
    /// Trace SmolLM2-135M-Instruct across all 11 stages (Prefill + 1-step Decode with KV Cache)
    Real,
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long, value_enum, default_value_t = Mode::Toy)]
    mode: Mode,

    /// Optional path to local Hugging Face model directory (config.json, tokenizer.json, model.safetensors).
    /// When omitted in `--mode real`, runs with SmolLM2-135M-Instruct exact architecture dimensions on deterministic synthetic weights.
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

#[derive(Clone, Copy)]
pub enum Role {
    Weight,     // Static parameter matrix: shape contains only (V, d, h, d_h, d_ff), never B or T
    Activation, // Dynamic intermediate activation: shape always contains batch B and sequence length T
    KvCache,    // Persisted historical K/V activation slice across autoregressive steps
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

fn default_model_type() -> String {
    "llama".to_string()
}

fn default_max_pos() -> usize {
    8192
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl LlamaConfig {
    /// Canonical configuration of `HuggingFaceTB/SmolLM2-135M-Instruct`
    pub fn smollm2_135m() -> Self {
        Self {
            model_type: "llama".to_string(),
            vocab_size: 49152,
            hidden_size: 576,
            intermediate_size: 1536,
            num_hidden_layers: 30,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 100_000.0,
            tie_word_embeddings: true,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

pub fn trace(stage: &str, name: &str, role: Role, t: &Tensor) -> candle_core::Result<()> {
    let role_tag = match role {
        Role::Weight => "WEIGHT    ",
        Role::Activation => "ACTIVATION",
        Role::KvCache => "KV_CACHE  ",
    };
    let elem_count = t.elem_count().min(4);
    let flat_slice = t.flatten_all()?.narrow(0, 0, elem_count)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
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

/// 2D-Reshape fast path for multiplying 3D activation [B, T, d_in] with safetensors weight [d_out, d_in]
pub fn linear_no_bias(x: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
    let (b, t, d_in) = x.dims3()?;
    let (d_out, _) = weight.dims2()?;
    x.reshape((b * t, d_in))?
        .matmul(&weight.t()?)?
        .reshape((b, t, d_out))
}

/// Explicit RMSNorm: x / sqrt(mean(x^2) + eps) * weight
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    candle_nn::ops::rms_norm(x, weight, eps as f32)
}

/// Precompute RoPE cos/sin tables of shape [max_seq_len, d_h / 2] for GPT-NeoX half-split rotary embedding
pub fn precompute_rope_tables(
    max_seq_len: usize,
    d_h: usize,
    theta: f32,
    dev: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let half = d_h / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / d_h as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
    let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
    let t = Tensor::from_vec(positions, (max_seq_len, 1), dev)?;
    let freqs = t.matmul(&inv_freq)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

/// GQA KV head expansion: [B, h_kv, T, d_h] -> [B, h_q, T, d_h]
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, h_kv, seq_len, d_h) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h_kv, n_rep, seq_len, d_h))?
        .reshape((b, h_kv * n_rep, seq_len, d_h))
}

/// Apply upper-triangular causal mask (j > i + offset set to -inf)
pub fn apply_causal_mask(scores: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
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

/// Verify the exact Toy MHA numbers from Section 5 of Episode 0
fn run_toy_verification(dev: &Device) -> Result<()> {
    println!("=== Episode 0 Toy Hand-Calculation Verification (B=1, T=3, d=4, h=2, d_h=2) ===");
    let (b, t, d, h, d_h) = (1, 3, 4, 2, 2);

    let x = Tensor::from_slice(
        &[
            1.0f32, 0.0, 1.0, 0.0, // t=0 ("我")
            0.0, 1.0, 1.0, 0.0, // t=1 ("喜欢")
            1.0, 1.0, 0.0, 1.0, // t=2 ("写")
        ],
        (b, t, d),
        dev,
    )?;
    trace("02.Embedding", "X (toy input states)", Role::Activation, &x)?;

    let w_q_math = Tensor::from_slice(
        &[
            1.0f32, 0.0, 0.0, 1.0, //
            0.0, 1.0, 1.0, 0.0, //
            1.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0,
        ],
        (d, d),
        dev,
    )?;
    let w_k_math = Tensor::from_slice(
        &[
            1.0f32, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0, //
            0.0, 1.0, 1.0, 0.0, //
            1.0, 0.0, 0.0, 1.0,
        ],
        (d, d),
        dev,
    )?;
    let w_v_math = Tensor::from_slice(
        &[
            1.0f32, 0.0, 2.0, 0.0, //
            0.0, 1.0, 0.0, 2.0, //
            1.0, 1.0, 0.0, 0.0, //
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

    let q = q_raw.reshape((b, t, h, d_h))?.transpose(1, 2)?.contiguous()?;
    let k = k_raw.reshape((b, t, h, d_h))?.transpose(1, 2)?.contiguous()?;
    let v = v_raw.reshape((b, t, h, d_h))?.transpose(1, 2)?.contiguous()?;

    let scale = 1.0f64 / (d_h as f64).sqrt();
    let raw_dots = q.matmul(&k.t()?)?;
    let scaled_scores = (&raw_dots * scale)?;
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

/// Build or load weight tensors for SmolLM2-135M-Instruct
fn load_or_init_weights(
    model_dir: Option<&Path>,
    cfg: &LlamaConfig,
    dev: &Device,
) -> Result<HashMap<String, Tensor>> {
    if let Some(dir) = model_dir {
        let st_path = dir.join("model.safetensors");
        println!("Loading real SafeTensors weights from: {}", st_path.display());
        let raw = candle_core::safetensors::load(&st_path, dev)
            .with_context(|| format!("Failed to load {}", st_path.display()))?;
        let mut f32_map = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            f32_map.insert(k, v.to_dtype(DType::F32)?);
        }
        println!("Loaded {} parameter tensors in F32.", f32_map.len());
        return Ok(f32_map);
    }

    println!(
        "No --model-dir supplied; initializing SmolLM2-135M-Instruct exact architecture shapes (V={}, d={}, L={}, h_q={}, h_kv={}, d_h={}, d_ff={}) with deterministic small weights...",
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        cfg.intermediate_size
    );
    let mut map = HashMap::new();
    let d = cfg.hidden_size;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
    let d_ff = cfg.intermediate_size;

    map.insert(
        "model.embed_tokens.weight".to_string(),
        (Tensor::ones((cfg.vocab_size, d), DType::F32, dev)? * 0.02)?,
    );
    map.insert(
        "model.norm.weight".to_string(),
        Tensor::ones((d,), DType::F32, dev)?,
    );
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        map.insert(format!("{p}.input_layernorm.weight"), Tensor::ones((d,), DType::F32, dev)?);
        map.insert(format!("{p}.self_attn.q_proj.weight"), (Tensor::ones((d, d), DType::F32, dev)? * 0.01)?);
        map.insert(format!("{p}.self_attn.k_proj.weight"), (Tensor::ones((kv_dim, d), DType::F32, dev)? * 0.01)?);
        map.insert(format!("{p}.self_attn.v_proj.weight"), (Tensor::ones((kv_dim, d), DType::F32, dev)? * 0.01)?);
        map.insert(format!("{p}.self_attn.o_proj.weight"), (Tensor::ones((d, d), DType::F32, dev)? * 0.01)?);
        map.insert(format!("{p}.post_attention_layernorm.weight"), Tensor::ones((d,), DType::F32, dev)?);
        map.insert(format!("{p}.mlp.gate_proj.weight"), (Tensor::ones((d_ff, d), DType::F32, dev)? * 0.01)?);
        map.insert(format!("{p}.mlp.up_proj.weight"), (Tensor::ones((d_ff, d), DType::F32, dev)? * 0.01)?);
        map.insert(format!("{p}.mlp.down_proj.weight"), (Tensor::ones((d, d_ff), DType::F32, dev)? * 0.01)?);
    }
    println!(
        "Initialized {} parameter tensors (tie_word_embeddings = {}).",
        map.len(),
        cfg.tie_word_embeddings
    );
    Ok(map)
}

/// Execute a single forward step (Prefill when T > 1, Decode when T = 1) and trace all 11 stages on Layer 0
fn forward_step(
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

    // Stage 1: Input Token IDs
    trace("01.Tokens", "input_ids", Role::Activation, input_ids)?;

    // Stage 2: Embedding lookup
    let w_embed = &weights["model.embed_tokens.weight"];
    trace("02.Embedding", "model.embed_tokens.weight", Role::Weight, w_embed)?;
    let flat_ids = input_ids.flatten_all()?;
    let mut hidden = w_embed.index_select(&flat_ids, 0)?.reshape((b, seq_len, d))?;
    trace("02.Embedding", "X_0 (embedded hidden state)", Role::Activation, &hidden)?;

    // Slice RoPE tables for current position window [offset .. offset + seq_len]
    let cos = cos_all.narrow(0, offset, seq_len)?;
    let sin = sin_all.narrow(0, offset, seq_len)?;

    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        let trace_layer = l == 0;

        let w_ln1 = &weights[&format!("{p}.input_layernorm.weight")];
        let x_norm = rms_norm(&hidden, w_ln1, cfg.rms_norm_eps)?;
        if trace_layer {
            trace("03.PreRMSNorm", &format!("{p}.input_layernorm.weight"), Role::Weight, w_ln1)?;
            trace("03.PreRMSNorm", "X_norm (Layer 0)", Role::Activation, &x_norm)?;
        }

        // Stage 4: QKV Projections & Head Split
        let w_q = &weights[&format!("{p}.self_attn.q_proj.weight")];
        let w_k = &weights[&format!("{p}.self_attn.k_proj.weight")];
        let w_v = &weights[&format!("{p}.self_attn.v_proj.weight")];
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

        // Stage 3/4: Apply RoPE to Q and K
        let q_rope = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_rope = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
        if trace_layer {
            trace("03.RoPE", "Q_rope (rotated queries)", Role::Activation, &q_rope)?;
            trace("03.RoPE", "K_rope (rotated keys)", Role::Activation, &k_rope)?;
        }

        // KV Cache Append along sequence axis (dim = 2)
        let (k_cache, v_cache) = match kv_caches[l].take() {
            Some((prev_k, prev_v)) => (Tensor::cat(&[&prev_k, &k_rope], 2)?, Tensor::cat(&[&prev_v, &v], 2)?),
            None => (k_rope, v),
        };
        if trace_layer {
            trace("05.KVCache", "Layer 0 K_cache", Role::KvCache, &k_cache)?;
            trace("05.KVCache", "Layer 0 V_cache", Role::KvCache, &v_cache)?;
        }
        kv_caches[l] = Some((k_cache.clone(), v_cache.clone()));

        // Stage 5: GQA Repeat KV & Scaled Dot-Product Attention
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

        // Stage 6: Merge Heads & Output Projection
        let o_concat = context.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, d))?;
        let w_o = &weights[&format!("{p}.self_attn.o_proj.weight")];
        let attn_out = linear_no_bias(&o_concat, w_o)?;
        let h_mid = (&hidden + &attn_out)?;
        if trace_layer {
            trace("06.MergeOut", "O_concat", Role::Activation, &o_concat)?;
            trace("06.MergeOut", &format!("{p}.self_attn.o_proj.weight"), Role::Weight, w_o)?;
            trace("06.MergeOut", "H_mid (after Attn residual)", Role::Activation, &h_mid)?;
        }

        // Stage 7: Post-Attention RMSNorm & SwiGLU MLP
        let w_ln2 = &weights[&format!("{p}.post_attention_layernorm.weight")];
        let h_norm = rms_norm(&h_mid, w_ln2, cfg.rms_norm_eps)?;
        let w_gate = &weights[&format!("{p}.mlp.gate_proj.weight")];
        let w_up = &weights[&format!("{p}.mlp.up_proj.weight")];
        let w_down = &weights[&format!("{p}.mlp.down_proj.weight")];

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

    // Stage 8: Final Layer Output + Final RMSNorm
    let w_final_norm = &weights["model.norm.weight"];
    let h_final = rms_norm(&hidden, w_final_norm, cfg.rms_norm_eps)?;
    trace("08.RepeatL30", "X_30 (after 30 layers + norm)", Role::Activation, &h_final)?;

    // Stage 9: Output Projection (LM Head tied to embed_tokens.weight)
    let w_lm_head = weights.get("lm_head.weight").unwrap_or(w_embed);
    trace("09.LM_Head", "lm_head.weight (tied W_E)", Role::Weight, w_lm_head)?;
    let logits = linear_no_bias(&h_final, w_lm_head)?;
    trace("09.LM_Head", "Logits [B, T, V]", Role::Activation, &logits)?;

    // Stage 10: Next-Token Selection at last sequence index (T - 1)
    let last_logits = logits.i((0, seq_len - 1))?;
    trace("10.SelectNext", "last_logits [V]", Role::Activation, &last_logits)?;
    let next_token_id = last_logits.argmax(0)?.to_scalar::<u32>()?;
    println!("[10.SelectNext ] Greedy argmax next_token_id = {}", next_token_id);

    Ok(next_token_id)
}

fn run_real_trace(model_dir: Option<&Path>, dev: &Device) -> Result<()> {
    let cfg = if let Some(dir) = model_dir {
        let cfg_str = std::fs::read_to_string(dir.join("config.json"))?;
        serde_json::from_str::<LlamaConfig>(&cfg_str)?
    } else {
        LlamaConfig::smollm2_135m()
    };

    let weights = load_or_init_weights(model_dir, &cfg, dev)?;
    let (cos_all, sin_all) = precompute_rope_tables(128, cfg.head_dim(), cfg.rope_theta, dev)?;
    let mut kv_caches: Vec<Option<(Tensor, Tensor)>> = vec![None; cfg.num_hidden_layers];

    let tokenizer = if let Some(dir) = model_dir {
        let tok_path = dir.join("tokenizer.json");
        if tok_path.exists() {
            Some(
                tokenizers::Tokenizer::from_file(&tok_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?,
            )
        } else {
            None
        }
    } else {
        None
    };

    let raw_prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    let prompt_ids: Vec<u32> = if let Some(ref tok) = tokenizer {
        let enc = tok
            .encode(raw_prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {e}"))?;
        println!(
            "\n[01.Tokenizer ] Raw prompt: {:?}\n[01.Tokenizer ] Tokens ({}): {:?}\n[01.Tokenizer ] Token IDs : {:?}",
            raw_prompt,
            enc.get_ids().len(),
            enc.get_tokens(),
            enc.get_ids()
        );
        enc.get_ids().to_vec()
    } else {
        vec![1, 4093, 198, 19556, 2, 198, 1, 520, 9531, 198]
    };
    let prefill_input = Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), dev)?;

    // Step 1: Prefill
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
    if let Some(ref tok) = tokenizer {
        if let Ok(piece) = tok.decode(&[first_generated_id], false) {
            println!("[10.Decoded    ] Step 1 generated token {} -> {:?}", first_generated_id, piece);
        }
    }

    // Step 2: Autoregressive Decode
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
    if let Some(ref tok) = tokenizer {
        if let Ok(piece) = tok.decode(&[second_generated_id], false) {
            println!("[10.Decoded    ] Step 2 generated token {} -> {:?}", second_generated_id, piece);
        }
        if let Ok(full) = tok.decode(&[first_generated_id, second_generated_id], false) {
            println!("[11.Summary    ] First 2 generated tokens decoded: {:?}", full);
        }
    }

    Ok(())
}

fn run_inspect(model_dir: Option<&Path>, dev: &Device) -> Result<()> {
    let cfg = if let Some(dir) = model_dir {
        let cfg_str = std::fs::read_to_string(dir.join("config.json"))?;
        serde_json::from_str::<LlamaConfig>(&cfg_str)?
    } else {
        LlamaConfig::smollm2_135m()
    };

    println!("\n=== [1/2] Blueprint Parsed from config.json ===");
    println!(
        "  model_type: {:?} | vocab_size (V): {} | hidden_size (d): {} | layers (L): {}",
        cfg.model_type, cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers
    );
    println!(
        "  attn_heads (h_q): {} | kv_heads (h_kv): {} | head_dim (d_h): {} | intermediate_size (d_ff): {}",
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        cfg.intermediate_size
    );
    println!(
        "  max_position_embeddings: {} | rope_theta: {:.0} | rms_norm_eps: {:e} | tie_word_embeddings: {}",
        cfg.max_position_embeddings, cfg.rope_theta, cfg.rms_norm_eps, cfg.tie_word_embeddings
    );

    let weights = load_or_init_weights(model_dir, &cfg, dev)?;

    println!("\n=== [2/2] SmolLM2-135M-Instruct SafeTensors Anatomy (1 + 30 * 9 + 1 = 272 Tensors) ===");
    let w_embed = &weights["model.embed_tokens.weight"];
    println!(
        "[Part 1: Shared Embedding] model.embed_tokens.weight            | shape: {:<14} | {:>10} params",
        format!("{:?}", w_embed.dims()),
        w_embed.elem_count()
    );

    let layer0_keys = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ];
    let mut layer0_params = 0usize;
    println!("[Part 2: Layer 0..29 (30 identical blocks, 9 tensors each = 270 tensors)]");
    for k in layer0_keys {
        let t = &weights[k];
        layer0_params += t.elem_count();
        println!(
            "  ├─ {:<44} | shape: {:<14} | {:>10} params",
            k,
            format!("{:?}", t.dims()),
            t.elem_count()
        );
    }
    println!(
        "  └─ Per-layer subtotal: {} params  x  {} layers = {} params",
        layer0_params,
        cfg.num_hidden_layers,
        layer0_params * cfg.num_hidden_layers
    );

    let w_norm = &weights["model.norm.weight"];
    println!(
        "[Part 3: Final RMSNorm   ] model.norm.weight                    | shape: {:<14} | {:>10} params",
        format!("{:?}", w_norm.dims()),
        w_norm.elem_count()
    );

    let total_params: usize = weights.values().map(|t| t.elem_count()).sum();
    let bf16_bytes = total_params * 2;
    println!("------------------------------------------------------------------------------------------");
    println!(
        "Total Tensors: {} | Total Parameters: {} (~134.52M) | BF16 Disk Footprint: {:.2} MB",
        weights.len(),
        total_params,
        bf16_bytes as f64 / 1_000_000.0
    );
    Ok(())
}

fn resolve_model_dir(user_dir: Option<&Path>) -> Option<PathBuf> {
    let default_repo_model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/SmolLM2-135M-Instruct");
    if let Some(dir) = user_dir {
        if dir.join("config.json").exists() {
            return Some(dir.to_path_buf());
        }
        let from_manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
        if from_manifest.join("config.json").exists() {
            return Some(from_manifest);
        }
    }
    if default_repo_model.join("config.json").exists() {
        return Some(default_repo_model);
    }
    user_dir.map(|p| p.to_path_buf())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = Device::Cpu;
    let resolved_dir = resolve_model_dir(args.model_dir.as_deref());
    match args.mode {
        Mode::Inspect => run_inspect(resolved_dir.as_deref(), &dev)?,
        Mode::Toy => run_toy_verification(&dev)?,
        Mode::Real => run_real_trace(resolved_dir.as_deref(), &dev)?,
    }
    Ok(())
}
