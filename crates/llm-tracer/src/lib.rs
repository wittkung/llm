//! `llm-tracer`: White-box tensor & forward-pass tracer for Episode 0.
//!
//! Organized into four single-responsibility modules following the reading order of the article:
//! - [`inspect`]: Section 2 (`--mode inspect`) — parses `config.json` and audits all 272 tensors in `model.safetensors`.
//! - [`toy`]:     Section 6 (`--mode toy`) — verifies the hand-calculated $B=1, T=3, d=4$ Multi-Head Attention example.
//! - [`ops`]:     Section 9 — `trace` printer and the 5 core mathematical operator primitives.
//! - [`model`]:   Sections 9 & 10 (`--mode real`) — 11-stage forward pass and KV-cached autoregressive decoding.

pub mod inspect;
pub mod model;
pub mod ops;
pub mod prelude;
pub mod toy;

pub use inspect::LlamaConfig;
pub use ops::{trace, Role};

#[cfg(test)]
mod tests {
    use crate::ops::{apply_causal_mask, linear_no_bias, repeat_kv};
    use crate::prelude::run_prelude_sandbox;
    use crate::toy::run_toy_verification;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_prelude_sandbox_runs_cleanly() {
        let dev = Device::Cpu;
        run_prelude_sandbox(&dev).expect("Prelude sandbox should succeed");
    }

    #[test]
    fn test_toy_verification_runs_cleanly() {
        let dev = Device::Cpu;
        run_toy_verification(&dev).expect("Toy MHA verification should succeed");
    }

    #[test]
    fn test_linear_no_bias_and_gqa_repeat_kv_shapes() {
        let dev = Device::Cpu;
        let x = Tensor::zeros((1, 10, 576), candle_core::DType::F32, &dev).unwrap();
        let w_k = Tensor::zeros((192, 576), candle_core::DType::F32, &dev).unwrap();
        let k_lin = linear_no_bias(&x, &w_k).unwrap();
        assert_eq!(k_lin.dims(), &[1, 10, 192]);

        let k_split = k_lin.reshape((1, 10, 3, 64)).unwrap().transpose(1, 2).unwrap();
        assert_eq!(k_split.dims(), &[1, 3, 10, 64]);

        let k_full = repeat_kv(&k_split, 3).unwrap();
        assert_eq!(k_full.dims(), &[1, 9, 10, 64]);
    }

    #[test]
    fn test_causal_mask_zeros_future_after_softmax() {
        let dev = Device::Cpu;
        let scores = Tensor::ones((1, 1, 3, 3), candle_core::DType::F32, &dev).unwrap();
        let masked = apply_causal_mask(&scores, 0).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&masked).unwrap();
        let rows = probs.squeeze(0).unwrap().squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        assert!((rows[0][0] - 1.0).abs() < 1e-6 && rows[0][1] == 0.0 && rows[0][2] == 0.0);
        assert!((rows[1][0] - 0.5).abs() < 1e-6 && (rows[1][1] - 0.5).abs() < 1e-6 && rows[1][2] == 0.0);
    }
}
