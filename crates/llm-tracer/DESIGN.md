# `llm-tracer`：Rust 白盒张量追踪器设计文档

本项目是《第零篇｜一次大语言模型推理的完整旅程》及后续篇章的配套 Rust 实验工程。

---

## 1. 为什么不用现成的 `candle-transformers` 封装？

`candle-transformers::models::llama` 或 `qwen2` 把整个前向过程封装在黑盒 `forward(&mut self, x: &Tensor, index_pos: usize)` 中。为了达到第零篇的验收标准——**“看清任意一个关键张量的维度、来源、流向，并严格区分静态参数矩阵（Weight）与动态中间激活（Activation）”**——我们需要一个**不足 250 行、完全展平、在每个算子之间内嵌 `TraceRecorder` 探针**的白盒实现。

---

## 2. 双模式 CLI 设计

通过一个统一的二进制入口支持文章中的两大核心实验：

```bash
# 实验 1：运行玩具维度手算验证（B=1, T=3, d=4, h=2, d_h=2, V=6）
# 完整打印每一个小矩阵的具体数值，与博客正文的手算过程逐格对应
cargo run -- --mode toy

# 实验 2：运行真实开源模型（如 SmolLM2-135M 或 Qwen2.5-0.5B）端到端追踪
# 打印静态权重清单 + Layer 0 全链路激活形状 + L1..L_{N-1} 残差流形状 + Prefill vs Decode 对比
cargo run -- --mode real --model-dir ./models/SmolLM2-135M-Instruct --prompt "Rust makes LLM tensors crystal clear" --steps 2
```

---

## 3. 核心探针数据结构（`probe.rs`）

探针系统強制区分 `TensorRole::Weight` 与 `TensorRole::Activation`，并自动校验：
- **凡是标记为 `Weight` 的张量**，其维度与输入序列长度 $T$ 无关；
- **凡是标记为 `Activation` 的张量**，高亮显示其中的 $[B, T, \dots]$ 或 $[B, h, T, \dots]$ 动态轴。

```rust
use candle_core::{Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    /// 训练固化的参数矩阵（来自 safetensors，普通推理中只读不更新）
    Weight,
    /// 随本次输入动态计算的中间激活（包含 B / T 维度）
    Activation,
    /// KV Cache 状态（历史激活的跨步持久化缓存）
    KvCache,
}

pub struct TraceEntry {
    pub stage_id: &'static str,   // 如 "04.AttnProj"
    pub symbol: &'static str,     // 对应博文数学符号，如 "Q_reshaped" 或 "W_Q"
    pub role: TensorRole,         // Weight / Activation / KvCache
    pub shape: Vec<usize>,        // 真实张量维度，如 [1, 9, 5, 64]
    pub formula_note: &'static str, // 简短来源说明，如 "reshape([B,T,h,d_h]).transpose(1,2)"
    pub preview: Vec<f32>,        // 前 4 个元素数值摘要
}
```

---

## 4. 模块划分规划

```text
crates/llm-tracer/
├── Cargo.toml
├── DESIGN.md
└── src/
    ├── main.rs          # CLI 参数解析与双模式调度
    ├── probe.rs         # 张量探针、Markdown/ASCII 表格格式化输出器
    ├── toy_manual.rs    # 配套示例 1：B=1, T=3, d=4, h=2, d_h=2, V=6 全量矩阵打印与验算
    └── real_model.rs    # 配套示例 2：加载真实 safetensors + tokenizer，执行 Layer 0 细粒度探针 + Prefill/Decode 两步追踪
```

---

## 5. 依赖选型（最小化依赖树）

在 `Cargo.toml` 中仅引入最核心、编译飞快的纯 Rust 组件：
- `candle-core = "0.8"`（或当前最新稳定版）：核心 Tensor 与 safetensors mmap 加载
- `candle-nn = "0.8"`：仅用于对照验证 `rms_norm` 与 `rotary_emb`
- `tokenizers = { version = "0.20", default-features = false, features = ["onig"] }`：分词与解码
- `serde` / `serde_json`：解析模型目录下的 `config.json`
- `anyhow` / `clap`：错误处理与命令行参数
