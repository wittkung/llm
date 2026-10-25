use anyhow::Result;
use candle_core::Device;
use clap::{Parser, ValueEnum};
use llm_tracer::{inspect, model, toy};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Mode {
    /// Mode 1 (Section 2): Inspect the 272 parameter tensors of SmolLM2-135M-Instruct
    Inspect,
    /// Mode 2 (Section 6): Verify the hand-calculated Toy MHA example (B=1, T=3, d=4, h=2, d_h=2)
    Toy,
    /// Mode 3 (Sections 9 & 10): Trace SmolLM2-135M-Instruct across all 11 stages (Prefill + Decode)
    Real,
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Execution mode: `inspect`, `toy`, or `real`
    #[arg(long, value_enum, default_value_t = Mode::Inspect)]
    mode: Mode,

    /// Path to the model directory containing `config.json`, `tokenizer.json`, and `model.safetensors`
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

/// Automatically locates `models/SmolLM2-135M-Instruct` relative to the workspace or current dir
fn resolve_model_dir(user_dir: Option<&Path>) -> PathBuf {
    let default_repo_model =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/SmolLM2-135M-Instruct");
    if let Some(dir) = user_dir {
        if dir.join("config.json").exists() {
            return dir.to_path_buf();
        }
        let from_manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
        if from_manifest.join("config.json").exists() {
            return from_manifest;
        }
        return dir.to_path_buf();
    }
    default_repo_model
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = Device::Cpu;
    let model_dir = resolve_model_dir(args.model_dir.as_deref());
    match args.mode {
        Mode::Inspect => inspect::run_inspect(&model_dir, &dev)?,
        Mode::Toy => toy::run_toy_verification(&dev)?,
        Mode::Real => model::run_real_trace(&model_dir, &dev)?,
    }
    Ok(())
}
