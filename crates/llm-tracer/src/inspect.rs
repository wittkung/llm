use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Maps directly to the architectural hyperparameters in `models/SmolLM2-135M-Instruct/config.json`
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub model_type: String,
    pub vocab_size: usize,              // V = 49,152
    pub hidden_size: usize,             // d = 576
    pub intermediate_size: usize,       // d_ff = 1,536
    pub num_hidden_layers: usize,       // L = 30
    pub num_attention_heads: usize,     // h_q = 9
    pub num_key_value_heads: usize,     // h_kv = 3
    pub max_position_embeddings: usize, // T_max = 8,192
    pub rms_norm_eps: f64,              // eps = 1e-5
    pub rope_theta: f32,                // theta = 100,000.0
    pub tie_word_embeddings: bool,      // true (shares embed_tokens with lm_head)
}

impl LlamaConfig {
    /// Single attention head dimension: d_h = d / h_q = 576 / 9 = 64
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Reads `config.json` and loads all 272 BF16 tensors from `model.safetensors` into F32 CPU tensors
pub fn load_config_and_weights(
    model_dir: &Path,
    dev: &Device,
) -> Result<(LlamaConfig, HashMap<String, Tensor>)> {
    let cfg_path = model_dir.join("config.json");
    let cfg_str = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("Failed to read {}", cfg_path.display()))?;
    let cfg: LlamaConfig = serde_json::from_str(&cfg_str)?;

    let st_path = model_dir.join("model.safetensors");
    println!("Loading real SafeTensors weights from: {}", st_path.display());
    let raw = candle_core::safetensors::load(&st_path, dev)
        .with_context(|| format!("Failed to load {}", st_path.display()))?;

    let mut f32_map = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        f32_map.insert(k, v.to_dtype(DType::F32)?);
    }
    println!("Loaded {} parameter tensors in F32.", f32_map.len());
    Ok((cfg, f32_map))
}

/// Safely fetches a named weight tensor from the loaded SafeTensors map without panicking
pub fn get_tensor<'a>(weights: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
    weights
        .get(key)
        .with_context(|| format!("Missing required weight tensor '{key}' in model.safetensors"))
}

/// Mode 1 (`--mode inspect`): Unpacks `config.json` and audits all 272 tensors in `model.safetensors`
pub fn run_inspect(model_dir: &Path, dev: &Device) -> Result<()> {
    let cfg_str = std::fs::read_to_string(model_dir.join("config.json"))?;
    let cfg: LlamaConfig = serde_json::from_str(&cfg_str)?;

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

    let (_, weights) = load_config_and_weights(model_dir, dev)?;

    println!("\n=== [2/2] SmolLM2-135M-Instruct SafeTensors Anatomy (1 + 30 * 9 + 1 = 272 Tensors) ===");
    let w_embed = get_tensor(&weights, "model.embed_tokens.weight")?;
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
        let t = get_tensor(&weights, k)?;
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

    let w_norm = get_tensor(&weights, "model.norm.weight")?;
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
