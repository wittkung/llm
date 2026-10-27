# 第零篇｜大模型推理全流程：从文本到下一个 Token

大语言模型在计算机内部究竟是什么？我们输入的一句文本，又是怎样穿过一层层参数矩阵、最终变成下一个输出词的？

作为本系列的总览篇，我们基于 **Rust 工具链**，以开源模型 `SmolLM2-135M-Instruct`（标准 Llama 架构，1.35 亿参数、272 个权重张量、体积 269 MB）为基准标本，完整追踪一次前向推理的 **11 个计算阶段**；并在文末对标工业界前沿模型 `DeepSeek-V4.1-Flash-1017` 与机械可解释性理论。

- **前置准备**：若你对矩阵维度消去律或 Rust 语法尚不熟悉，建议先阅读 [《前置篇｜线性代数与 Rust 极简入门》](../ep-pre-math-and-rust/draft.md)。
- **配套仓库**：[https://github.com/wittkung/llm](https://github.com/wittkung/llm)

---

## 一、 为什么用 Rust 追踪大模型？

仅需五行 Python 代码即可调用开源模型完成对话，但高度封装的接口也掩盖了底层的张量变形与内存流转。

而在工业界生产环境中，大模型基础设施的核心组件早已转向 Rust：

- **终端与编辑器工具**：OpenAI 官方 CLI `Codex`、AI 编辑器 `Zed`，以及 `Claude Code` 与 `Cursor` 的底层代码检索引擎均由 Rust 编写。
- **Python 生态底层内核**：日常调用的 `tokenizers` 分词器、`safetensors` 权重加载器、`Pydantic` 校验核心与 `uv` 包管理器，底层全部是 Rust 实现。

### 1. 零 GC 开销与编译期内存安全

大模型推理涉及高频分词、几十 GB 权重的零拷贝内存映射（`mmap`）以及高并发流式调度。

Python 受限于解释器开销与全局解释器锁（GIL）；C/C++ 则容易引入内存越界与多线程数据竞争。Rust 通过编译期**所有权**（Ownership）与**借用检查**（Borrow Checker），在不依赖垃圾回收（GC）的前提下，从编译阶段消除了内存与并发安全隐患。

### 2. 基于 LLVM 的原生跨平台编译

Rust 无需像 Python 或 Java 那样在目标机器上配置庞大的解释器或虚拟环境，其跨平台能力建立在三层编译器机制之上：

- **LLVM 中间表示**（LLVM IR）：编译器前端 `rustc` 先将源码转化为硬件无关的 LLVM IR，再由 LLVM 后端直接针对 Apple Silicon（`aarch64`）、x86_64 或 WebAssembly（`wasm32`）生成原生机器码。
- **编译期条件编译**（`#[cfg(...)]`）：可在编译阶段按操作系统自动挂载硬件加速库（如 macOS 自动链接 `Accelerate` 矩阵框架，Linux/Windows 自动切换通用 CPU 或 CUDA 算子）。
- **静态链接单文件分发**：所有依赖默认静态编译为单一原生可执行文件，彻底告别动态库缺失与环境冲突。

---

## 二、 环境准备与模型文件拆解

本实验无需安装 Python 或 Conda，仅需配置 Rust 工具链并下载三个模型文件。

### 1. 安装 Rust 工具链

**macOS / Linux / WSL 终端：**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
cargo --version
```

- 第 1 行拉取官方安装脚本，出现提示时按回车（Enter）选择默认安装；
- 第 2 行刷新当前终端的环境变量路径；
- 第 3 行验证构建系统与包管理器 `cargo` 是否就绪。

**Windows PowerShell：**

```powershell
winget install Rustlang.Rustup
```

若系统未安装 Visual Studio C++ 链接器，安装器会自动提示 `1) Quick install via the Visual Studio Community installer`，直接按回车自动配置即可。完成后重开 PowerShell 窗口运行 `cargo --version`。

### 2. 三个核心模型文件

Hugging Face（[huggingface.co](https://huggingface.co)）是开源模型权重的标准托管平台。在 `HuggingFaceTB/SmolLM2-135M-Instruct` 仓库中，驱动推理只需下载三个核心文件：`config.json`、`tokenizer.json` 与 `model.safetensors`。

#### （1）config.json：架构超参数

`config.json` 以纯文本记录了决定网络拓扑与张量形状的全部**架构超参数**（Architectural Hyperparameters）：

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "vocab_size": 49152,
  "hidden_size": 576,
  "num_hidden_layers": 30,
  "num_attention_heads": 9,
  "num_key_value_heads": 3,
  "intermediate_size": 1536,
  "hidden_act": "silu",
  "max_position_embeddings": 8192,
  "rope_theta": 100000,
  "rms_norm_eps": 1e-05,
  "tie_word_embeddings": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "torch_dtype": "bfloat16"
}
```

各字段对应的数学符号与所作用的推理阶段如下表所示（阅读后文各阶段时可随时回查）：

| JSON 字段 | 符号与数值 | 作用阶段 | 核心含义 |
|---|---|---|---|
| `vocab_size` | $V = 49,152$ | 阶段 01, 02, 09 | 词表词元总数（Token ID 范围 `0..49151`） |
| `hidden_size` | $d = 576$ | 全流程主干 | 隐藏状态（残差流）的特征向量维度 |
| `num_hidden_layers` | $L = 30$ | 阶段 08 | 串联堆叠的 Transformer 层数 |
| `num_attention_heads` | $h_q = 9$ | 阶段 04 — 06 | 查询（Query）注意力头数，单头维度 $d_h = 576 / 9 = 64$ |
| `num_key_value_heads` | $h_{kv} = 3$ | 阶段 04 — 06 | 键值（Key/Value）头数，构成 3:1 分组查询注意力（GQA） |
| `intermediate_size` | $d_{ff} = 1536$ | 阶段 07 | SwiGLU 前馈网络的中间升维宽度 |
| `rope_theta` | $\theta = 100,000$ | 阶段 03 | 旋转位置编码（RoPE）的几何级数基频 |
| `rms_norm_eps` | $\epsilon = 10^{-5}$ | 阶段 07, 08 | RMSNorm 分母防除零平滑常数 |
| `tie_word_embeddings` | `true` | 阶段 02, 09 | 入口词嵌入矩阵与出口 LM Head 共享同一张权重表 |
| `torch_dtype` | `bfloat16` | 磁盘存储 | 权重采用 16 位脑浮点数（每参数占 2 字节） |

围绕 `"hidden_size": 576`，有两个关键设计逻辑：

- **分布式表示**（Distributed Representation）：离散的整数词号无法直接求导或表达语义距离；将每个词映射为 576 维连续实数向量后，神经网络即可通过线性代数运算度量词法、句法与语境特征。
- **硬件寄存器对齐**：576 在注意力层被均分为 $h_q = 9$ 个头，单头维度 $d_h = 576 \div 9 = 64$。现代 CPU SIMD 与 GPU Tensor Core 在维度为 $32, 64, 128$ 等 **$2$ 的整数次幂**时计算吞吐率最高。

#### （2）tokenizer.json：子词分词器与词表

为什么输入文本要先做**子词分词**（Subword Tokenization）？

若按整句编号，人类句子无穷无尽，词表无法穷举；若按单字符切分，不仅单字符缺乏独立语义，还会使序列长度 $T$ 暴涨，导致复杂度为 $O(T^2)$ 的注意力计算崩溃。

字节对编码（BPE）采取了折中路线：统计语料高频片段，将常用整词、词根词缀与基础字符浓缩为一张 $V = 49,152$ 的有限词表：

```json
{
  "model": {
    "type": "BPE",
    "vocab": {
      "<|endoftext|>": 0,
      "<|im_start|>": 1,
      "<|im_end|>": 2,
      "!": 17,
      "Ċ": 198,
      "user": 4093,
      "Hello": 19556,
      "ĠRust": 36828,
      "ectable": 49151
    },
    "merges": ["Ġ t", "Ġ a", "i n", "h e", "Ġ Ġ", "r e"]
  }
}
```

> **词号与向量的分工**：`tokenizer.json` 仅维护 **`字符串 <-> 整数行号（Token ID）`** 的双向字典（体积仅 2.0 MB），并不存储那 576 个浮点数。真正的浮点数向量存储在 `model.safetensors` 的第一张矩阵 `model.embed_tokens.weight`（形状 `[49152, 576]`）中——整数 Token ID 就是去该矩阵抽取值向量的**行下标**。

#### （3）model.safetensors：静态参数张量

`model.safetensors` 以二进制格式存储了模型全部 **272 个静态参数张量**（共 `134,515,008` 个浮点数参数）。文件头部是记录张量名称、数据类型与字节偏移量的 JSON 索引，紧接着是按 64 字节对齐的原始内存块，支持通过 `mmap` 零拷贝映射进内存。

围绕这份权重文件，有三个核心问题需要先建立直觉：

**第一，为什么采用 `BF16` 浮点数格式？**

日常代码中的 32 位单精度浮点数（`FP32`）每参数占 4 字节，显存开销过大；而传统 16 位半精度浮点数（`FP16`，1 位符号 + 5 位指数 + 10 位尾数）由于指数位仅有 5 位（最大表示 $65504$），在多层矩阵连乘时极易发生数值溢出（`Inf/NaN`）。

现代大模型普遍采用 **BF16**（Brain Floating Point 16，每参数占 **2 字节**）：它完整保留了与 `FP32` 相同的 **8 位指数位**（1 位符号 + 8 位指数 + 7 位尾数），以牺牲末尾几位尾数精度的微小代价，换取了与 32 位浮点数一致的抗溢出动态范围，并将磁盘与显存体积直接减半：

$$
134,515,008\text{ 参数} \times 2\text{ Bytes} = 269,030,016\text{ Bytes} \approx \mathbf{269.03\text{ MB}}
$$

**第二，第一张矩阵 `model.embed_tokens.weight` 有多大？**

词表文件 `tokenizer.json` 仅存文本与整数行号，体积只有 **2.0 MB**。而紧接着它的第一张权重矩阵——**词嵌入矩阵**（Token Embedding Matrix，记作 $\mathbf{W}_E \in \mathbb{R}^{V \times d}$），形状为 `[49152, 576]`，单张矩阵就包含：

$$
49,152\text{ 行} \times 576\text{ 列} = 28,311,552\text{ 个参数} \quad (\text{占全模型总参数的 } \mathbf{21.05\%}\text{，物理体积 } \mathbf{56.62\text{ MB}})
$$

它的第 `0` 行至第 `49151` 行，与 `tokenizer.json` 中的 `49,152` 个 Token ID 严格一一对应。

**第三，这些浮点数参数是怎么定出来的？推理时还会变吗？**

- **随机初始化**（训练第 0 步）：模型刚创建时，所有参数仅由均值为 $0$、标准差为 $1/\sqrt{d} \approx 0.04167$ 的正态分布随机数生成器填满，此时每行向量只是随机白噪声。
- **预训练与反向传播**（训练期持续更新）：在数万亿词的语料上，模型不断执行“根据上文预测下一个词”的任务，计算预测分布与真实下一词之间的**交叉熵损失**（Cross-Entropy Loss），再通过**反向传播**（Backpropagation）计算每个参数的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ 并由优化器微调数值。经常出现在相似语境中的词（如 `"Rust"` 与 `"C++"`），其对应的 576 维向量会在梯度驱动下在几何空间中彼此靠拢。
- **推理阶段**（永久冻结、只读不改）：一旦训练完成并导出为 `model.safetensors`，全部 272 个张量即作为**静态权重**永久冻结。本地推理时，模型仅从表中只读拷贝数据，绝不会修改权重文件中的任何数字。

> **为什么不用下载仓库里的其他文件？**
> - `tokenizer_config.json` / `special_tokens_map.json`：特殊控制词元 ID 已内嵌于 `tokenizer.json` 中，本篇直接在代码中明文书写 ChatML 模板字符串以保持白盒透明；
> - `generation_config.json`：记录随机采样的 `temperature` 与 `top_p` 参数；本篇采用 100% 确定性的**贪心解码**（Greedy Decoding）以确保跨平台数值完全可复现；
> - `vocab.json` / `merges.txt`：旧版 Python 分词器的拆分文件，数据已完全包含在 `tokenizer.json` 内。

### 3. 克隆仓库并下载模型

**macOS / Linux / WSL 终端：**

```bash
git clone https://github.com/wittkung/llm.git
cd llm
mkdir -p models/SmolLM2-135M-Instruct

BASE_URL="https://hf-mirror.com/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main"
curl -L "${BASE_URL}/config.json"       -o models/SmolLM2-135M-Instruct/config.json
curl -L "${BASE_URL}/tokenizer.json"    -o models/SmolLM2-135M-Instruct/tokenizer.json
curl -L "${BASE_URL}/model.safetensors" -o models/SmolLM2-135M-Instruct/model.safetensors
```

**Windows PowerShell：**

```powershell
git clone https://github.com/wittkung/llm.git
cd llm
New-Item -ItemType Directory -Force -Path models\SmolLM2-135M-Instruct | Out-Null

$BASE_URL = "https://hf-mirror.com/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main"
curl.exe -L "$BASE_URL/config.json"       -o models\SmolLM2-135M-Instruct\config.json
curl.exe -L "$BASE_URL/tokenizer.json"    -o models\SmolLM2-135M-Instruct\tokenizer.json
curl.exe -L "$BASE_URL/model.safetensors" -o models\SmolLM2-135M-Instruct\model.safetensors
```

### 4. 运行 llm-tracer 开箱体检

配套仓库中的 `crates/llm-tracer` 是专为本篇编写的白盒探针包，各源码模块的职责划分如下：

| 源码文件 | 行数 | 对应章节 | 运行命令 | 核心职责 |
|---|---|---|---|---|
| `src/main.rs` & `lib.rs` | 59 / 62 | 全局入口 | `cargo test` | 命令行参数解析与单元测试 |
| `src/prelude.rs` | 150 | [前置篇](../ep-pre-math-and-rust/draft.md) | `cargo run -- --mode prelude` | 6 组线性代数与张量算子演练 |
| `src/inspect.rs` | 139 | 第二节 | `cargo run -- --mode inspect` | 解析配置并体检 272 个权重张量 |
| `src/toy.rs` | 83 | 第六节 | `cargo run -- --mode toy` | 核验 $3 \times 4$ 玩具注意力手算结果 |
| `src/ops.rs` | 104 | 第九节 | （由 `model.rs` 调用） | `trace` 打印器与 5 个核心数学算子 |
| `src/model.rs` | 249 | 第九、十节 | `cargo run -- --mode real` | 11 阶段真机前向推理与 KV Cache |

`src/inspect.rs` 负责反序列化 `config.json`，并将 `model.safetensors` 中的 272 个 `BF16` 张量转换为 CPU 原生支持的 32 位单精度浮点数 `F32`：

```rust
// crates/llm-tracer/src/inspect.rs（核心逻辑）
pub fn load_config_and_weights(
    model_dir: &Path,
    dev: &Device,
) -> Result<(LlamaConfig, HashMap<String, Tensor>)> {
    let cfg_str = std::fs::read_to_string(model_dir.join("config.json"))?;
    let cfg: LlamaConfig = serde_json::from_str(&cfg_str)?;

    let raw = candle_core::safetensors::load(model_dir.join("model.safetensors"), dev)?;
    let mut f32_map = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        f32_map.insert(k, v.to_dtype(DType::F32)?);
    }
    Ok((cfg, f32_map))
}
```

在终端运行 `--mode inspect`（注意中间的 `--` 用于将右侧参数传递给 `llm-tracer`）：

```bash
cargo run -- --mode inspect
```

终端随即输出完整的模型权重解剖报告：

```text
=== [1/2] Blueprint Parsed from config.json ===
  model_type: "llama" | vocab_size (V): 49152 | hidden_size (d): 576 | layers (L): 30
  attn_heads (h_q): 9 | kv_heads (h_kv): 3 | head_dim (d_h): 64 | intermediate_size (d_ff): 1536
  max_position_embeddings: 8192 | rope_theta: 100000 | rms_norm_eps: 1e-5 | tie_word_embeddings: true
Loading real SafeTensors weights from: ../../models/SmolLM2-135M-Instruct/model.safetensors
Loaded 272 parameter tensors in F32.

=== [2/2] SmolLM2-135M-Instruct SafeTensors Anatomy (1 + 30 * 9 + 1 = 272 Tensors) ===
[Part 1: Shared Embedding] model.embed_tokens.weight            | shape: [49152, 576]   |   28311552 params
[Part 2: Layer 0..29 (30 identical blocks, 9 tensors each = 270 tensors)]
  ├─ model.layers.0.input_layernorm.weight        | shape: [576]          |        576 params
  ├─ model.layers.0.self_attn.q_proj.weight       | shape: [576, 576]     |     331776 params
  ├─ model.layers.0.self_attn.k_proj.weight       | shape: [192, 576]     |     110592 params
  ├─ model.layers.0.self_attn.v_proj.weight       | shape: [192, 576]     |     110592 params
  ├─ model.layers.0.self_attn.o_proj.weight       | shape: [576, 576]     |     331776 params
  ├─ model.layers.0.post_attention_layernorm.weight | shape: [576]          |        576 params
  ├─ model.layers.0.mlp.gate_proj.weight          | shape: [1536, 576]    |     884736 params
  ├─ model.layers.0.mlp.up_proj.weight            | shape: [1536, 576]    |     884736 params
  ├─ model.layers.0.mlp.down_proj.weight          | shape: [576, 1536]    |     884736 params
  └─ Per-layer subtotal: 3540096 params  x  30 layers = 106202880 params
[Part 3: Final RMSNorm   ] model.norm.weight                    | shape: [576]          |        576 params
------------------------------------------------------------------------------------------
Total Tensors: 272 | Total Parameters: 134515008 (~134.52M) | BF16 Disk Footprint: 269.03 MB
```

### 5. 权重结构拆解：1 + 30 × 9 + 1 = 272

272 个张量并非杂乱无章，而是严格按 **`1 + 30 × 9 + 1`** 的三段式结构排布（共 211 张二维矩阵 + 61 条一维归一化向量）：

| 部位 | 张量数 | 核心权重与形状 | 参数量 | 占比 |
|---|---|---|---|---|
| **Part 1：共享词嵌入表** | $1$ | `embed_tokens.weight`: `[49152, 576]` | $28,311,552$ | $21.05\%$ |
| **Part 2：30 层 Transformer** | $30 \times 9 = 270$ | 每层 5 个注意力张量 + 4 个 MLP 张量 | $106,202,880$ | $78.95\%$ |
| **Part 3：最终归一化** | $1$ | `model.norm.weight`: `[576]` | $576$ | $0.0004\%$ |
| **全模型总计** | **$272$** | **211 张二维矩阵 + 61 条一维向量** | **$134,515,008$** | **$100\%$** |

30 层 Transformer（`model.layers.0` 至 `model.layers.29`）的内部结构与形状 100% 相同，每一层固定由 **9 个张量**（约 354 万参数）串联组成：

#### （1）自注意力子层（`self_attn`，占 5 个张量，共 88.5 万参数）

负责跨序列位置检索并聚合上下文信息：

- `input_layernorm.weight`（`[576]`）：注意力计算前的 Pre-RMSNorm 一维缩放向量；
- `q_proj.weight`（`[576, 576]`）：将输入投影为 $h_q = 9$ 个查询头（$9 \times 64 = 576$）；
- `k_proj.weight` 与 `v_proj.weight`（均为 `[192, 576]`）：将输入投影为 $h_{kv} = 3$ 个键值头（$3 \times 64 = 192$）；
- `o_proj.weight`（`[576, 576]`）：将多头注意力输出重新融合回 576 维残差流。

#### （2）前馈网络子层（`mlp`，占 4 个张量，共 265.4 万参数）

负责单位置内部的非线性特征变换与知识提取（占单层参数量的 **75%**，是注意力层的 3 倍）：

- `post_attention_layernorm.weight`（`[576]`）：进入 MLP 前的 Pre-RMSNorm 一维缩放向量；
- `gate_proj.weight` 与 `up_proj.weight`（均为 `[1536, 576]`）：将 576 维向量双路升维至 $d_{ff} = 1536$ 维；
- `down_proj.weight`（`[576, 1536]`）：将门控调制后的 1536 维特征投影回 576 维残差流。

> **核心结论**：在计算机磁盘中，大语言模型既没有 `if-else` 逻辑规则，也没有压缩文本库；它的本体就是一个包含 **272 个命名浮点数数组**（共计 1.345 亿个小数） 的静态数据字典。

---

## 三、 参数矩阵与中间激活

当模型执行前向计算（Forward Pass）时，内存中的所有张量严格分为两类：

1. **参数矩阵**（Parameter Matrices / Weights）：从 `.safetensors` 加载，训练期定型，推理期全程**只读不变**。
2. **中间激活**（Intermediate Activations）：随当前输入文本动态计算生成，推理结束后（除 KV Cache 外）立即释放。

区分两者的唯一标准是**维度签名**。设批大小为 $B$、序列长度为 $T$、隐藏维度为 $d$、头数为 $h$、单头维度为 $d_h$、词表大小为 $V$：

> **维度判定铁律**：所有**参数矩阵**的形状仅由架构常数 $(V, d, h, d_h, d_{ff})$ 决定，绝不包含 $B$ 和 $T$；所有**中间激活**的形状必定包含批大小 $B$ 与序列长度 $T$。

整趟前向计算的 **11 阶段张量流转全景** 如下表所示：

| 阶段 | 形状变化 | 静态参数矩阵 | 动态中间激活 |
|---|---|---|---|
| **01. 输入与分词** | 文本 $\to [B, T]$ | 词表（`tokenizer.json`） | `input_ids`: $[B, T]$ |
| **02. Embedding 查表** | $[B, T] \to [B, T, d]$ | $\mathbf{W}_E: [V, d]$ | $\mathbf{H}^{(0)}: [B, T, d]$ |
| **03. 位置编码**（RoPE） | 保持 $[B, h, T, d_h]$ | 预计算频率表 $[T_{\max}, d_h/2]$ | $\mathbf{Q}_{\text{rope}}, \mathbf{K}_{\text{rope}}: [B, h, T, d_h]$ |
| **04. QKV 投影与拆头** | $[B, T, d] \to [B, h, T, d_h]$ | $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ | $\mathbf{Q}, \mathbf{K}, \mathbf{V}: [B, h, T, d_h]$ |
| **05. 注意力与掩码** | $[B, h, T, d_h] \to [B, h, T, T] \to [B, h, T, d_h]$ | 无（缩放常数与因果掩码） | $\mathbf{S}, \mathbf{A}: [B, h, T, T]$，$\mathbf{O}_h: [B, h, T, d_h]$ |
| **06. 多头合并与投影** | $[B, h, T, d_h] \to [B, T, d]$ | $\mathbf{W}_O: [d, d]$ | $\mathbf{O}_{\text{concat}}, \Delta\mathbf{H}_{\text{attn}}: [B, T, d]$ |
| **07. 归一化、残差与 MLP** | $[B, T, d] \to [B, T, d_{ff}] \to [B, T, d]$ | $\boldsymbol{\gamma}: [d]$，$\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{down}}$ | $\mathbf{G}, \mathbf{U}: [B, T, d_{ff}]$，$\mathbf{H}^{(1)}: [B, T, d]$ |
| **08. 串联多层 Transformer** | $\mathbf{H}^{(0)} \to \dots \to \mathbf{H}^{(L)}: [B, T, d]$ | 第 $0 \dots L-1$ 层权重组 | $\mathbf{H}^{(\ell)}: [B, T, d]$ |
| **09. 输出投影**（LM Head） | $[B, T, d] \to [B, T, V]$ | $\mathbf{W}_{\text{head}}: [V, d]$ | $\text{logits}_{\text{all}}: [B, T, V]$ |
| **10. 选出下一个 Token** | $[B, T, V] \to [B, V] \to [B, 1]$ | 无参数矩阵 | $\text{logits}_{\text{last}}: [B, V]$ |
| **11. 自回归生成循环** | 追加新词，步进循环 | 全模型权重复用 | 新增激活与 **KV Cache** |

---

## 四、 前向计算入口（阶段 01 — 03）

### 01. 输入与分词（Tokenization）

用户输入 `"Hello"` 后，客户端会先按对话模板（ChatML）包裹控制符与换行符：

```text
<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n
```

分词器查 `tokenizer.json` 将该字符串切分为 $T = 10$ 个整数 Token ID：

| 位置下标 $t$ | $0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ | $9$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **切分词元** | `<\|im_start\|>` | `user` | `\n` (`Ċ`) | `Hello` | `<\|im_end\|>` | `\n` (`Ċ`) | `<\|im_start\|>` | `ass` | `istant` | `\n` (`Ċ`) |
| **Token ID** | `1` | `4093` | `198` | `19556` | `2` | `198` | `1` | `520` | `9531` | `198` |

> **为什么 `"assistant"` 被切成了 `"ass"`（520）+ `"istant"`（9531）？**
> 英文正文中该词前面通常带空格（`"Ġassistant"` 在词表中是独立的 `11191` 号整词），而模板中 `<|im_start|>assistant` 紧挨特殊控制符无空格，分词器便按 BPE 规则将其拆分为两个已知词根。

为便于后续手算演练，我们同时定义一个 $B=1, T=3$ 的极简序列 `["我", "喜欢", "写"]`（对应 ID `[1, 3, 4]`）：

$$
\text{input\_ids} = \begin{bmatrix} 1 & 3 & 4 \end{bmatrix}, \qquad \text{Shape: } [B, T] = [1, 3]
$$

### 02. Embedding 查表

为什么不能把 `input_ids = [1, 3, 4]` 直接送进神经网络做矩阵乘法？

因为整数编号 `1`、`3`、`4` 只是词典里的离散门牌号，本身没有任何代数意义——编号 `4`（`"写"`）绝不等于编号 `1`（`"我"`）加编号 `3`（`"喜欢"`）。要让模型计算词与词之间的语义关联，必须先把每个离散的整数编号映射为一条包含 $d$ 个浮点数的连续向量（在 `SmolLM2-135M` 中 $d = 576$）。

#### （1）按行号从词嵌入矩阵 $\mathbf{W}_E$ 抽取值向量

这项工作由我们在第二节认识的第一张权重矩阵——**词嵌入矩阵** $\mathbf{W}_E \in \mathbb{R}^{V \times d}$（`model.embed_tokens.weight`，形状 `[49152, 576]`）完成。

当形状为 `[B=1, T=3]` 的 `input_ids = [[1, 3, 4]]` 到达时，代码调用 `w_embed.index_select(&flat_ids, 0)`，直接把每个整数 ID 当作**行号**，去 $\mathbf{W}_E$ 中把对应的那一行（长 $d$ 的浮点数向量）整行抽出来：

```text
词嵌入矩阵 W_E (形状 [V, d])                 抽出的 3 行叠成 H^(0) (形状 [B=1, T=3, d])
┌──────────────────────────────┐
│ 第 0 行: [ 0.1,  0.2, ... ]  │
│ 第 1 行: [ 1.0,  0.0, ... ]  │ ──(取 ID=1)──►  t=0 ("我"):   [ 1.0,  0.0, ... ] (长 d)
│ 第 2 行: [ 0.5, -0.3, ... ]  │
│ 第 3 行: [ 0.0,  1.0, ... ]  │ ──(取 ID=3)──►  t=1 ("喜欢"): [ 0.0,  1.0, ... ] (长 d)
│ 第 4 行: [ 1.0,  1.0, ... ]  │ ──(取 ID=4)──►  t=2 ("写"):   [ 1.0,  1.0, ... ] (长 d)
│ ...      ...                 │
└──────────────────────────────┘
```

把抽出来的 $T = 3$ 行按词语先后顺序上下叠在一起，再在外层套上批大小 $B = 1$ 的括号，原来的每个整数就展开成了一条长度为 $d$ 的向量，张量形状完成了第一次跃迁：

$$
\text{input\_ids}: [B, T] \xrightarrow{\text{按行查表 } \mathbf{W}_E: [V, d]} \mathbf{H}^{(0)}: [B, T, d]
$$

> **与独热向量（One-Hot）乘法的等价性**：在数学教材中，这一步常写为将整数 ID 转为仅第 $\text{id}$ 位为 $1$、其余全为 $0$ 的 $V$ 维独热向量 $\mathbf{e}_{\text{id}} \in \{0,1\}^{1 \times V}$，再右乘 $\mathbf{W}_E \in \mathbb{R}^{V \times d}$。工程代码中直接按行下标拷贝内存（`index_select`），省去了 $49,151$ 次乘零运算，数学结果完全等价。

#### （2）什么是隐藏状态（Hidden States）与残差流（Residual Stream）？

查表所得的三维张量 $\mathbf{H}^{(0)}: [B, T, d]$ 称为第 0 层初始**隐藏状态**（Hidden States）。

在 Transformer 架构中，这个形状恒定为 $[B, T, d]$ 的主干张量又被称为**残差流**（Residual Stream）：你可以把它看作一条从第 0 层直通第 30 层出口的主干通道——接下来的每一层注意力子层与 MLP 子层，都只是从这条主干道上读取当前的 $\mathbf{H}^{(\ell)}$，算出一个形状同为 $[B, T, d]$ 的修正增量 $\Delta\mathbf{H}$，然后通过**逐元素加法**（$\mathbf{H} \leftarrow \mathbf{H} + \Delta\mathbf{H}$）叠加回这条主干道上。

---

### 03. 旋转位置编码（RoPE）

#### （1）为什么矩阵运算需要额外注入位置信息？

观察刚刚查表得到的 $\mathbf{H}^{(0)}: [B, T, d]$：第 $0$ 行是第 1 个词的向量，第 $1$ 行是第 2 个词的向量。

在接下来的网络层中，线性投影与前馈网络都是**逐行独立计算**的，而注意力机制的向量点积也只关心“两个向量的内容长什么样”，本身并不知道它们排在第几行。

举个直观的例子：如果把输入句子从 `"猫 追 狗"` 调换成 `"狗 追 猫"`，查表得到的 $\mathbf{H}^{(0)}$ 无非是把第 $0$ 行和第 $2$ 行对调了位置。如果不注入位置序号 $t \in \{0, 1, \dots, T-1\}$，后续所有矩阵算出来的结果也只是原封不动地对调第 $0$ 行和第 $2$ 行——模型根本分不清到底是谁追谁。

#### （2）绝对位置编码 vs. 现代旋转位置编码（RoPE）

怎么把位置序号 $t$ 告诉模型？两代架构采用了截然不同的思路：

- **早期绝对位置编码**（如 GPT-2）：在阶段 02 查表结束后，立刻给第 $t$ 行的词向量直接加上一条代表“第 $t$ 号座位”的固定位置向量 $\mathbf{p}_t \in \mathbb{R}^d$（即 $\mathbf{H}^{(0)}_t \leftarrow \mathbf{H}^{(0)}_t + \mathbf{p}_t$）。
  **缺陷**：自然语言中最核心的是词与词之间的**相对距离**（比如形容词紧挨在名词前面 1 个位置），而不是绝对座位号。如果硬加绝对位置向量，同一个词组出现在第 2 个词和出现在第 500 个词时，向量数值完全不同，模型很难泛化到长文本。
- **现代旋转位置编码**（Rotary Position Embedding, **RoPE**，如 Llama、Qwen、SmolLM2）：在阶段 02 查表后**不修改 $\mathbf{H}^{(0)}$**；而是等到下一阶段（阶段 04）生成注意力查询向量 $\mathbf{Q}$ 和键向量 $\mathbf{K}$ 时，根据词元所在的位置下标 $t$，把向量在平面上**旋转一个与 $t$ 成正比的角度 $t\theta$**！

#### （3）RoPE 是怎样旋转一个 64 维向量的？（分三步拆解）

我们在前置篇第三节学过，二维平面旋转矩阵 $\mathbf{R}(\alpha) = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix}$ 每次只能旋转一个包含 **2 个数字**的二维向量 $[x^{(1)}, x^{(2)}]$。

而在 `SmolLM2-135M` 中，每个注意力头的查询向量 $\mathbf{q}$ 和键向量 $\mathbf{k}$ 长度为 **$d_h = 64$**。怎么旋转一个 64 维向量？RoPE 的做法非常巧妙，分三步完成：

**第一步：两两配对，拆成 32 组二维坐标**（半区配对）

把长度为 $64$ 的向量 $[x_0, x_1, \dots, x_{63}]$ 前半区（下标 $0 \dots 31$）和后半区（下标 $32 \dots 63$）一一配对，分成 **32 组独立的二维向量**：

- 第 $0$ 组：$\begin{bmatrix} x_0 \\ x_{32} \end{bmatrix}$，第 $1$ 组：$\begin{bmatrix} x_1 \\ x_{33} \end{bmatrix}$，$\dots$，第 $i$ 组：$\begin{bmatrix} x_i \\ x_{i+32} \end{bmatrix}$，$\dots$，第 $31$ 组：$\begin{bmatrix} x_{31} \\ x_{63} \end{bmatrix}$。

**第二步：按位置下标 $t$ 成倍旋转**（第 $t$ 个词就转 $t$ 倍角度）

给第 $i$ 组二维向量设定一个固定的“单步旋转角” $\omega_i$（称为**基准角频率**）。当词元处在序列的第 $t$ 个位置（$t = 0, 1, 2, \dots$）时，就把这组二维向量旋转 **$t \cdot \omega_i$** 弧度：

- **位置 $t = 0$**（第 1 个词）：旋转 $0 \cdot \omega_i = 0$ 弧度（保持原样不转）；
- **位置 $t = 1$**（第 2 个词）：旋转 $1 \cdot \omega_i$ 弧度；
- **位置 $t = 2$**（第 3 个词）：旋转 $2 \cdot \omega_i$ 弧度；
- **通用位置 $t$**：乘上旋转角为 $t\omega_i$ 的二维正交旋转矩阵 $\mathbf{R}(t\omega_i)$：

$$
\begin{bmatrix} x'_i \\ x'_{i+32} \end{bmatrix} =
\underbrace{\begin{bmatrix} \cos(t\omega_i) & -\sin(t\omega_i) \\ \sin(t\omega_i) & \cos(t\omega_i) \end{bmatrix}}_{\mathbf{R}(t\omega_i)}
\begin{bmatrix} x_i \\ x_{i+32} \end{bmatrix}
=
\begin{bmatrix}
x_i \cos(t\omega_i) - x_{i+32} \sin(t\omega_i) \\
x_i \sin(t\omega_i) + x_{i+32} \cos(t\omega_i)
\end{bmatrix}
$$

**第三步：32 组采用快慢不同的旋转速度**（高低频多尺度表盘）

为什么 32 组二维向量不干脆用同一个旋转步长 $\omega$，而要按几何级数公式 $\omega_i = \frac{1}{100000^{2i / 64}}$（其中 `100000` 即 `config.json` 中的 `rope_theta`）让每一组的旋转步长越来越慢？

这就好比钟表必须同时拥有**秒针、分针和时针**：

- **第 $0$ 组**（高频“秒针”，$i=0$）：$\omega_0 = \frac{1}{100000^0} = \mathbf{1.0}$ 弧度/词（每往后走 1 个词就大幅旋转约 $57.3^\circ$）。它对相邻一两个词的先后次序极其敏感，但每过 $6$ 个词就会转满一圈（$360^\circ$）发生角度重叠；
- **中间组**（中频“分针”）：旋转速度适中，负责分辨几十到几百个词之间的中距离结构；
- **第 $31$ 组**（超低频“时针”，$i=31$）：$\omega_{31} = \frac{1}{100000^{62/64}} \approx \mathbf{0.000014}$ 弧度/词。它转得极慢，哪怕横跨 $8,192$ 个词的上下文窗口也不会转满一圈，专门负责感知长文档的宏观远近距离。

#### （4）为什么这样旋转就能让模型只看“相对距离 $(m - n)$”？

这正是我们在前置篇第三节证明过的**正交旋转矩阵点积保角性**！

当序列位置 $m$ 的查询向量 $\mathbf{q}_m$（已旋转 $m\omega_i$ 角）与位置 $n$ 的键向量 $\mathbf{k}_n$（已旋转 $n\omega_i$ 角）在注意力层做点积时，两个旋转矩阵合二为一：

$$
(\mathbf{q}_m \mathbf{R}(m\omega_i)) \cdot (\mathbf{k}_n \mathbf{R}(n\omega_i)) = \mathbf{q}_m \, \mathbf{R}(m\omega_i)\mathbf{R}(n\omega_i)^\top \, \mathbf{k}_n^\top = \mathbf{q}_m \, \mathbf{R}\big((m - n)\omega_i\big) \, \mathbf{k}_n^\top
$$

你看！绝对位置 $m$ 和 $n$ 自动抵消成了它们的差值 **$(m - n)$**！

这意味着：无论 `"喜欢"` 和 `"写"` 这两个词出现在句首的位置 $(1, 2)$，还是出现在长文末尾的位置 $(501, 502)$，只要它们一前一后相隔 $1$ 个词（$m - n = 1$），它们旋转后的夹角差就恒等于 $1 \cdot \omega_i$，算出来的注意力匹配分数完全一致！这就是 RoPE 统治现代开源大模型的数学原因。

---

## 五、 注意力机制（阶段 04 — 06）

在初始隐藏状态 $\mathbf{H}^{(0)}$ 中，每一行仅包含当前词自身的字面特征。**多头自注意力**（Multi-Head Self-Attention） 的作用是让每个位置根据特征匹配度，动态检索并加权融合前文所有位置的信息。

```mermaid
flowchart TD
    H_in["输入隐藏状态 H_in (动态激活)<br/>Shape: [B, T, d]"]

    subgraph Stage04["阶段 04：注意力线性投影与拆头 (满足 d = h × d_h)"]
        WQ["参数矩阵 W_Q: [d, d]"]
        WK["参数矩阵 W_K: [d, d]"]
        WV["参数矩阵 W_V: [d, d]"]

        Q_proj["Q: [B, h, T, d_h]"]
        K_proj["K: [B, h, T, d_h]"]
        V_proj["V: [B, h, T, d_h]"]
    end

    H_in --> WQ --> Q_proj
    H_in --> WK --> K_proj
    H_in --> WV --> V_proj

    subgraph Stage05["阶段 05：缩放点积注意力与因果掩码"]
        KT["K 转置最后两维 K^T: [B, h, d_h, T]"]
        Scores["注意力分数 S = QK^T / √d_h: [B, h, T, T]"]
        Mask["施加因果掩码 (上三角 j > i 置为 -∞)"]
        Probs["注意力权重 A = Softmax(S_masked): [B, h, T, T]"]
        OutH["上下文聚合 O_h = A @ V: [B, h, T, d_h]"]
    end

    Q_proj --> Scores
    K_proj --> KT --> Scores
    Scores --> Mask --> Probs
    Probs --> OutH
    V_proj --> OutH

    subgraph Stage06["阶段 06：多头合并与输出投影"]
        Concat["Transpose(1,2) + Reshape 合并多头<br/>→ O_concat: [B, T, d]"]
        WO["参数矩阵 W_O: [d, d]"]
        DeltaH["注意力层输出 ΔH_attn: [B, T, d]"]
    end

    OutH --> Concat --> WO --> DeltaH
```

### 04. QKV 线性投影与拆头

#### （1）为什么要投影出 Q、K、V 三个矩阵，而不能直接拿 $\mathbf{H}$ 和自己做点积？

在进入注意力层时，我们手头只有一张隐藏状态 $\mathbf{H}: [B, T, d]$。如果直接拿 $\mathbf{H}$ 和它自己的转置做点积（$\mathbf{H}\mathbf{H}^\top$），由于任何向量永远和它自己一模一样（夹角为 $0^\circ$），算出来的最高分永远在对角线上——每个词都只会死盯着自己看！

而在真实语言里，上下文检索是**非对称**的：比如当读到代词 `"它"` 时，`"它"` 需要去前面寻找一个**名词**（如 `"手机"`），而不是寻找另一个 `"它"`。

为了实现这种“提问者”与“被检索者”的分离，模型用三张独立训练的参数矩阵 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$，把同一个输入隐藏状态 $\mathbf{H}$ 投影成三份职责截然不同的中间激活：

- **查询向量**（Query, $\mathbf{Q}$）：当前位置发出的检索条件——代表“我正在前文中寻找什么特征？”；
- **键向量**（Key, $\mathbf{K}$）：每个位置对外展示的特征标签——代表“我自身具备什么语法或语义属性可供匹配？”；
- **值向量**（Value, $\mathbf{V}$）：一旦 $\mathbf{Q}$ 与 $\mathbf{K}$ 配对成功，该位置实际交出来供汇总融合的语义内容向量。

$$
\mathbf{Q}_{\text{raw}} = \mathbf{H}\mathbf{W}_Q^\top, \quad \mathbf{K}_{\text{raw}} = \mathbf{H}\mathbf{W}_K^\top, \quad \mathbf{V}_{\text{raw}} = \mathbf{H}\mathbf{W}_V^\top
$$

> **为什么公式里写的是转置 $\mathbf{W}^\top$？**
> 在 Hugging Face `.safetensors` 文件中，线性层权重统一按 `[输出维度, 输入维度]` 存放（如 `k_proj.weight` 形状为 `[192, 576]`）。当形状为 `[B, T, 576]` 的激活右乘权重时，必须先将权重转置为 `[576, 192]`，才能消去内侧维度 `576` 得到 `[B, T, 192]`。

#### （2）为什么要拆分注意力头（Split Heads）？怎么拆？

一句话里往往同时存在多条线索：既有动词找宾语的语法关系，又有代词找先行词的指代关系。如果把 $d = 576$ 维捆在一起只算一组注意力概率，多种关系就会在一次加权平均里互相混杂。

因此，模型把总特征维度拆分成多个独立的**注意力头**（Attention Heads，每个头的子空间宽度固定为 $d_h = 64$），让每个头在自己的 64 维子空间里独立计算一套注意力权重。

在代码中，“拆头”不需要乘任何参数矩阵，直接用我们在前置篇第五节学过的 **`reshape` + `transpose(1, 2)`** 两步完成（以 $h_q = 9$ 个查询头为例）：

1. **第一步 `reshape`**（把 576 拆成 $9 \times 64$）：将 `[B, T, 576]` 重塑为四维张量 `[B, T, 9, 64]`（代表 $B$ 个批次、$T$ 个词、每个词有 $9$ 个头、每头 $64$ 维）；
2. **第二步 `transpose(1, 2)`**（把“头维度”换到前面）：交换第 1 维（$T$）和第 2 维（$9$），形状变成 **`[B, 9, T, 64]`**——这样最内侧的两维就变成了干净的 `[T, 64]` 二维矩阵，使 9 个注意力头可以各自并行计算 `[T, 64] × [64, T] -> [T, T]`！

$$
[B, T, h \cdot d_h] \xrightarrow{\text{1. Reshape 拆维}} [B, T, h, d_h] \xrightarrow{\text{2. Transpose}(1,2)\text{ 换轴}} [B, h, T, d_h]
$$

拆成 `[B, h, T, 64]` 之后，代码紧接着对 $\mathbf{Q}$ 和 $\mathbf{K}$ 调用阶段 03 介绍的 **RoPE 旋转位置编码**（注意：值向量 $\mathbf{V}$ 不参与计算匹配分数，因此**只旋转 $\mathbf{Q}$ 和 $\mathbf{K}$，不旋转 $\mathbf{V}$**）。

#### （3）什么是分组查询注意力（GQA）？（为什么 Q 是 9 个头，K 和 V 只有 3 个头？）

在早期的标准多头注意力（MHA）中，查询头、键头、值头的数量完全相等（都是 9 个头）。但在推理生成长文时，每一层都要把历史所有词的 $\mathbf{K}$ 和 $\mathbf{V}$ 缓存在显存里（即第十节的 KV Cache），9 个键值头非常占显存。

为了给显存减负，`SmolLM2-135M` 采用了现代大模型标配的**分组查询注意力**（Grouped-Query Attention, **GQA**）：

- **查询头保留 $h_q = 9$ 个**：`q_proj.weight` 形状为 `[576, 576]`（$9 \times 64 = 576$），拆头后 $\mathbf{Q}$ 形状为 **`[B, 9, T, 64]`**；
- **键值头缩减为 $h_{kv} = 3$ 个**：`k_proj.weight` 与 `v_proj.weight` 形状只有 **`[192, 576]`**（$3 \times 64 = 192$），拆头后 $\mathbf{K}, \mathbf{V}$ 形状为 **`[B, 3, T, 64]`**（显存直接省去三分之二！）；
- **3:1 共享广播**（`repeat_kv`）：在进入阶段 05 做点积前，程序调用我们在前置篇第五节学过的 `unsqueeze -> expand -> reshape` 把 3 个键值头各自原地复制 3 份（第 `0,1,2` 号查询头共享第 `0` 号键值头，第 `3,4,5` 号共享第 `1` 号，第 `6,7,8` 号共享第 `2` 号），将 $\mathbf{K}, \mathbf{V}$ 广播对齐为 **`[B, 9, T, 64]`**。

---

### 05. 缩放点积注意力与因果掩码

现在，9 个注意力头各自拿到了一份形状为 $[T, d_h]$（$d_h = 64$）的 $\mathbf{Q}$、$\mathbf{K}$ 和 $\mathbf{V}$。注意力机制的计算按顺序分四步走：

#### 第一步：计算 QK 点积匹配分数（`[T, 64] × [64, T] -> [T, T]`）

把键张量 $\mathbf{K}: [B, h, T, d_h]$ 的最后两维转置为 $\mathbf{K}^\top: [B, h, d_h, T]$，与查询张量 $\mathbf{Q}: [B, h, T, d_h]$ 相乘。内侧的单头维度 $d_h = 64$ 在点积求和中消去，产出一个 $T \times T$ 的方阵：

$$
\mathbf{Q}\mathbf{K}^\top : [B, h, T, d_h] \times [B, h, d_h, T] \longrightarrow [B, h, T, T]
$$

在这个 $T \times T$ 方阵中，**第 $i$ 行、第 $j$ 列的数字，就代表“第 $i$ 个词的查询向量 $\mathbf{q}_i$”与“第 $j$ 个词的键向量 $\mathbf{k}_j$”之间的点积相似度**。

#### 第二步：除以缩放因子 $\sqrt{d_h}$

当单头维度 $d_h = 64$ 时，64 个乘积加在一起会让点积数值的方差放大 64 倍（标准差放大 $\sqrt{64} = 8$ 倍）。如果点积数值过大，下一步求指数 $e^x$ 时就会导致最大值垄断近 100% 的权重、其余位置梯度趋近于 $0$（称为 Softmax 饱和）。

因此，必须将点积矩阵的每个元素除以常数 $\sqrt{d_h} = \sqrt{64} = 8$，将数值拉回温和区间：

$$
\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}, \qquad \text{Shape: } [B, h, T, T]
$$

#### 第三步：施加因果掩码（Causal Mask，严禁偷看未来）

自回归语言模型有一条铁律：**因果性**（Causality）——当模型在第 $i$ 个词的位置预测下一个词时，它只能看到位置 $j \le i$ 的前文和自己，**绝对不能偷看位置 $j > i$ 的未来词元**！

以 $T = 3$ 个词（$t=0, 1, 2$）为例，在 $3 \times 3$ 的分数方阵中：

- 左下角及对角线（$j \le i$）是合法的历史与当前位置，掩码值为 **`0`**（加上 $0$ 不改变原分数）；
- 右上角严格上三角区域（$j > i$，即第 0 行看第 1、2 列，第 1 行看第 2 列）属于“偷看未来”，掩码矩阵将其设为**负无穷 $-\infty$**：

$$
\text{mask}_{[3 \times 3]} = \begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}, \qquad \mathbf{S}_{\text{masked}} = \mathbf{S} + \text{mask}
$$

#### 第四步：按行 Softmax 归一化并乘值矩阵 $\mathbf{V}$

对 $\mathbf{S}_{\text{masked}}$ 的**每一行**独立执行我们在前置篇第六节学过的 $\operatorname{softmax}$ 归一化。由于右上角未来位置全是 $-\infty$，取指数后 $e^{-\infty} = \mathbf{0}$，未来位置的权重被彻底归零，而每一行合法历史位置的权重之和严格等于 $1.0$：

$$
\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}} + \text{mask}\right), \qquad \text{Shape: } [B, h, T, T]
$$

最后，拿这张每行和为 $1$ 的注意力权重方阵 $\mathbf{A}: [B, h, T, T]$ 左乘值矩阵 $\mathbf{V}: [B, h, T, d_h]$。内侧的维度 $T$ 消去，尾部的 $d_h$ 重新浮现，把前文各位置的 $\mathbf{V}$ 向量按注意力权重加权融合到了当前位置上：

$$
\mathbf{O}_h = \mathbf{A}\mathbf{V} : [B, h, T, T] \times [B, h, T, d_h] \longrightarrow [B, h, T, d_h]
$$

---

### 06. 多头合并与输出投影

现在，$h = 9$ 个注意力头各自算出了自己的上下文聚合结果 $\mathbf{O}_h: [B, 9, T, 64]$。怎样把它们合回残差流主干所需的 $[B, T, 576]$？

只需把阶段 04 的拆头动作反过来执行，再乘一张融合矩阵：

1. **换轴拼回 `[B, T, 576]`**：先调 `.transpose(1, 2)?.contiguous()?` 把形状变回 `[B, T, 9, 64]`，再调 `.reshape((b, t, 576))?` 将每个词的 9 个 64 维子向量首尾拼接成一条 576 维向量 $\mathbf{O}_{\text{concat}}: [B, T, 576]$；
2. **右乘输出投影矩阵 $\mathbf{W}_O^\top$**：单纯拼接后，前 64 维只来自第 0 个头，第 64..128 维只来自第 1 个头，各头之间还没有交换信息。因此最后必须右乘本层第 4 张注意力参数矩阵 $\mathbf{W}_O^\top \in \mathbb{R}^{576 \times 576}$（`self_attn.o_proj.weight`），让 9 个头提取到的特征充分线性混合，交出注意力子层的最终增量 $\Delta\mathbf{H}_{\text{attn}}$：

$$
\Delta\mathbf{H}_{\text{attn}} = \mathbf{O}_{\text{concat}} \mathbf{W}_O^\top, \qquad \text{Shape: } [B, T, d] \times [d, d] \longrightarrow [B, T, d]
$$

---

## 六、 玩具维度手算示例

为彻底验证阶段 04 — 06 的矩阵计算，我们设定一组可纸笔手算的极简维度：$B = 1$、序列长度 $T = 3$（输入 `["我", "喜欢", "写"]`）、隐藏维度 $d = 4$、头数 $h = 2$、单头维度 $d_h = 2$。

### 1. 输入隐藏状态与 QKV 投影

设阶段 02 查表所得的输入激活矩阵 $\mathbf{X} \in \mathbb{R}^{3 \times 4}$ 为：

$$
\mathbf{X}_{[3 \times 4]} = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 1 & 0 \\
1 & 1 & 0 & 1
\end{bmatrix}
\begin{array}{l}
\leftarrow t=0 \text{ ("我")} \\
\leftarrow t=1 \text{ ("喜欢")} \\
\leftarrow t=2 \text{ ("写")}
\end{array}
$$

给定三张 $4 \times 4$ 参数矩阵（前 2 列对应 Head 1，后 2 列对应 Head 2）：

$$
\mathbf{W}_Q = \begin{bmatrix}
1 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1
\end{bmatrix}, \quad
\mathbf{W}_K = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
0 & 1 & 1 & 0 \\
1 & 0 & 0 & 1
\end{bmatrix}, \quad
\mathbf{W}_V = \begin{bmatrix}
1 & 0 & 2 & 0 \\
0 & 1 & 0 & 2 \\
1 & 1 & 0 & 0 \\
0 & 0 & 1 & 1
\end{bmatrix}
$$

计算矩阵乘法 $\mathbf{X}\mathbf{W}_Q, \mathbf{X}\mathbf{W}_K, \mathbf{X}\mathbf{W}_V$：

$$
\mathbf{Q}_{\text{raw}} = \begin{bmatrix}
2 & 0 & 1 & 1 \\
1 & 1 & 2 & 0 \\
1 & 2 & 1 & 2
\end{bmatrix}, \quad
\mathbf{K}_{\text{raw}} = \begin{bmatrix}
1 & 1 & 2 & 0 \\
0 & 2 & 1 & 1 \\
2 & 1 & 1 & 2
\end{bmatrix}, \quad
\mathbf{V}_{\text{raw}} = \begin{bmatrix}
2 & 1 & 2 & 0 \\
1 & 2 & 0 & 2 \\
1 & 1 & 3 & 3
\end{bmatrix}
$$

将其按列拆分为 **Head 1**（左半 $3 \times 2$）与 **Head 2**（右半 $3 \times 2$）：

$$
\text{Head 1: } \mathbf{Q}^{(1)} = \begin{bmatrix} 2 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix}, \;
\mathbf{K}^{(1)} = \begin{bmatrix} 1 & 1 \\ 0 & 2 \\ 2 & 1 \end{bmatrix}, \;
\mathbf{V}^{(1)} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \\ 1 & 1 \end{bmatrix}
$$

$$
\text{Head 2: } \mathbf{Q}^{(2)} = \begin{bmatrix} 1 & 1 \\ 2 & 0 \\ 1 & 2 \end{bmatrix}, \;
\mathbf{K}^{(2)} = \begin{bmatrix} 2 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix}, \;
\mathbf{V}^{(2)} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 3 & 3 \end{bmatrix}
$$

### 2. 注意力分数、掩码与加权求和

先计算 **Head 1** 的点积 $\mathbf{Q}^{(1)}(\mathbf{K}^{(1)})^\top$，除以 $\sqrt{2} \approx 1.414$ 并施加因果掩码（将 $j > i$ 置为 $-\infty$）：

$$
\mathbf{Q}^{(1)}(\mathbf{K}^{(1)})^\top = \begin{bmatrix}
2 & 0 & 4 \\
2 & 2 & 3 \\
3 & 4 & 4
\end{bmatrix}
\xrightarrow{\div\sqrt{2} + \text{mask}}
\mathbf{S}_{\text{masked}}^{(1)} \approx \begin{bmatrix}
1.414 & -\infty & -\infty \\
1.414 & 1.414 & -\infty \\
2.121 & 2.828 & 2.828
\end{bmatrix}
$$

对 $\mathbf{S}_{\text{masked}}^{(1)}$ 逐行求 $\operatorname{softmax}$：

- **第 0 行**（`"我"`）：仅第 0 列有效，概率必为 `[1.0, 0.0, 0.0]`；
- **第 1 行**（`"喜欢"`）：前两列数值相等，概率平分为 `[0.5, 0.5, 0.0]`；
- **第 2 行**（`"写"`）：三列经指数归一化后约为 `[0.20, 0.40, 0.40]`。

用所得权重方阵 $\mathbf{A}^{(1)}$ 左乘 $\mathbf{V}^{(1)}$：

$$
\mathbf{O}^{(1)} = \mathbf{A}^{(1)}\mathbf{V}^{(1)} =
\begin{bmatrix}
1.0 & 0.0 & 0.0 \\
0.5 & 0.5 & 0.0 \\
0.2 & 0.4 & 0.4
\end{bmatrix}
\begin{bmatrix}
2 & 1 \\
1 & 2 \\
1 & 1
\end{bmatrix}
= \begin{bmatrix}
2.0 & 1.0 \\
1.5 & 1.5 \\
1.2 & 1.4
\end{bmatrix}
$$

同理计算 **Head 2**：

$$
\mathbf{Q}^{(2)}(\mathbf{K}^{(2)})^\top = \begin{bmatrix}
2 & 2 & 3 \\
4 & 2 & 2 \\
2 & 3 & 5
\end{bmatrix}
\xrightarrow{\div\sqrt{2} + \text{mask}}
\begin{bmatrix}
\sqrt{2} & -\infty & -\infty \\
2\sqrt{2} & \sqrt{2} & -\infty \\
\sqrt{2} & \frac{3}{\sqrt{2}} & \frac{5}{\sqrt{2}}
\end{bmatrix}
$$

$$
\mathbf{A}^{(2)} \approx \begin{bmatrix}
1.00 & 0.00 & 0.00 \\
0.80 & 0.20 & 0.00 \\
0.09 & 0.18 & 0.73
\end{bmatrix}, \qquad
\mathbf{O}^{(2)} = \mathbf{A}^{(2)}\mathbf{V}^{(2)} \approx
\begin{bmatrix}
2.00 & 0.00 \\
1.61 & 0.39 \\
2.38 & 2.56
\end{bmatrix}
$$

对比第 1 行（`"喜欢"`）：Head 1 均等融合前两个词（`[0.5, 0.5, 0.0]`），而 Head 2 将 80% 的注意力集中在主语 `"我"` 上（`[0.80, 0.20, 0.00]`），直观体现了多头子空间的语义分工。

### 3. 多头拼接与代码核验

将 $\mathbf{O}^{(1)}$ 与 $\mathbf{O}^{(2)}$ 左右拼接回 $3 \times 4$ 矩阵：

$$
\mathbf{O}_{\text{concat}} = \begin{bmatrix} \mathbf{O}^{(1)} & \mathbf{O}^{(2)} \end{bmatrix} \approx
\begin{bmatrix}
2.00 & 1.00 & 2.00 & 0.00 \\
1.50 & 1.50 & 1.61 & 0.39 \\
1.20 & 1.40 & 2.38 & 2.56
\end{bmatrix}
$$

打开 `crates/llm-tracer/src/toy.rs`，可以看到上述四步手算过程在 Rust 中的直接映射：

```rust
// crates/llm-tracer/src/toy.rs（核心计算片段）
let q = q_raw.reshape((b, t, h, d_h))?.transpose(1, 2)?.contiguous()?;
let k = k_raw.reshape((b, t, h, d_h))?.transpose(1, 2)?.contiguous()?;
let v = v_raw.reshape((b, t, h, d_h))?.transpose(1, 2)?.contiguous()?;

let scale = 1.0f64 / (d_h as f64).sqrt();
let scaled_scores = (&q.matmul(&k.t()?)? * scale)?;
let masked_scores = apply_causal_mask(&scaled_scores, 0)?;
let attn_probs = candle_nn::ops::softmax_last_dim(&masked_scores)?;
let context = attn_probs.matmul(&v)?;
let o_concat = context.transpose(1, 2)?.contiguous()?.reshape((b, t, d))?;
```

运行 `cargo run -- --mode toy`，终端输出与纸笔手算结果完全一致：

```text
=== Episode 0 Toy Hand-Calculation Verification (B=1, T=3, d=4, h=2, d_h=2) ===
--- Head 1 Attention Probs A^(1) [3x3] ---
[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.19777581, 0.40111208, 0.40111208]]
--- Head 2 Attention Probs A^(2) [3x3] ---
[[1.0, 0.0, 0.0], [0.8044297, 0.19557032, 0.0], [0.087948754, 0.17837015, 0.73368114]]
--- Concatenated Output O_concat [3x4] ---
[[2.0, 1.0, 2.0, 0.0], [1.5, 1.5, 1.6088594, 0.39114064], [1.1977758, 1.4011121, 2.376941, 2.5577836]]
```

---

## 七、 残差流、MLP 与词表打分（阶段 07 — 11）

### 07. 残差连接、Pre-RMSNorm 与 SwiGLU MLP

#### （1）第一次残差相加与 Pre-RMSNorm 归一化

注意力子层算出上下文增量 $\Delta\mathbf{H}_{\text{attn}}: [B, T, d]$ 后，模型不会用它覆盖掉原来的输入 $\mathbf{H}^{(0)}$，而是执行一次**逐元素加法**（即**残差连接** Residual Connection）：

$$
\mathbf{H}_{\text{mid}} = \mathbf{H}^{(0)} + \Delta\mathbf{H}_{\text{attn}}, \qquad \text{Shape: } [B, T, d]
$$

为什么要用加法而不是直接替换？因为当网络堆叠到 30 层甚至上百层时，如果每一层都直接覆写张量，底层的原始词义特征与反向传播梯度很容易在连乘中消失；而通过残差加法 $\mathbf{H} + \Delta\mathbf{H}$，主干残差流始终畅通无阻，每一层只需专注于计算“当前层需要补充的修正量 $\Delta\mathbf{H}$”。

同时要注意归一化的位置：现代 Llama / SmolLM2 架构统一采用 **Pre-RMSNorm**（前置均方根归一化）——即保持主干残差流 $\mathbf{H}$ 原样不变，仅在**进入自注意力子层之前**（`input_layernorm`）和**进入 MLP 子层之前**（`post_attention_layernorm`）对分支输入做均方根归一化（公式见前置篇第六节）：

$$
\mathbf{X}_{\text{norm}} = \operatorname{RMSNorm}(\mathbf{H}_{\text{mid}}) = \frac{\mathbf{H}_{\text{mid}}}{\sqrt{\frac{1}{d}\sum_{i=1}^d H_{\text{mid}, i}^2 + \epsilon}} \odot \boldsymbol{\gamma}, \qquad \text{Shape: } [B, T, d]
$$

#### （2）SwiGLU 前馈网络（MLP）的三步非线性变换

如果说刚才的注意力层负责的是**跨位置搬运上下文信息**（让位置 $t$ 吸收位置 $0 \dots t-1$ 的特征），那么接下来的**多层感知机**（MLP / Feed-Forward Network）负责的就是**单位置内部的特征非线性加工与知识提取**——在 MLP 内部，不同位置 $t$ 之间没有任何交互，计算完全在每个词自己的 576 维向量上独立进行。

现代大模型普遍采用包含三张权重矩阵（`gate_proj`、`up_proj`、`down_proj`）的 **SwiGLU**（Swish-Gated Linear Unit）结构，分三步对 $\mathbf{X}_{\text{norm}}: [B, T, 576]$ 做变换：

**第一步：双路并联升维投影**（`[B, T, 576] -> [B, T, 1536]`）

把 576 维向量同时送入两条并联支路，各自右乘一张形状为 `[1536, 576]` 的转置矩阵，将特征空间撑宽至近 3 倍的中间维度 $d_{ff} = 1536$：

- **候选特征支路**（Up Projection）：$\mathbf{U} = \mathbf{X}_{\text{norm}}\mathbf{W}_{\text{up}}^\top$，形状为 `[B, T, 1536]`；
- **门控开关支路**（Gate Projection + SiLU 激活）：先乘 $\mathbf{W}_{\text{gate}}^\top$ 得到 `[B, T, 1536]`，再逐元素通过我们在前置篇第六节学过的 $\operatorname{SiLU}(x) = \frac{x}{1 + e^{-x}}$ 非线性激活函数，得到门控张量 $\mathbf{G} = \operatorname{SiLU}(\mathbf{X}_{\text{norm}}\mathbf{W}_{\text{gate}}^\top)$。

**第二步：Hadamard 逐元素门控相乘**（`[B, T, 1536]`）

将门控支路 $\mathbf{G}$ 与候选特征支路 $\mathbf{U}$ 做**逐元素乘法**（$\mathbf{G} \odot \mathbf{U}$，代码中为 `(&gate * &up)?`）：当门控张量 $\mathbf{G}$ 的某个维度接近 $0$ 时，就像开关一样把 $\mathbf{U}$ 对应维度的特征过滤关断；当 $\mathbf{G}$ 数值较大时则予以放行放大。

**第三步：降维还原并写回残差流**（`[B, T, 1536] -> [B, T, 576]`）

将门控调制后的 1536 维张量右乘第三张矩阵 $\mathbf{W}_{\text{down}}^\top$（`down_proj.weight`，形状 `[576, 1536]` 转置为 `[1536, 576]`），压缩回主干宽度 `[B, T, 576]`，并执行本层的第二次残差相加：

$$
\Delta\mathbf{H}_{\text{mlp}} = \big(\operatorname{SiLU}(\mathbf{X}_{\text{norm}}\mathbf{W}_{\text{gate}}^\top) \odot (\mathbf{X}_{\text{norm}}\mathbf{W}_{\text{up}}^\top)\big)\mathbf{W}_{\text{down}}^\top, \qquad \mathbf{H}^{(1)} = \mathbf{H}_{\text{mid}} + \Delta\mathbf{H}_{\text{mlp}}
$$

至此，第 0 层 Transformer 全部计算完毕：输入是 $\mathbf{H}^{(0)}: [B, T, 576]$，输出是融合了第 0 层上下文与非线性特征的 $\mathbf{H}^{(1)}: [B, T, 576]$。

---

### 08. 串联多层 Transformer

正因为单层 Transformer 严格遵守“输入 $[B, T, d]$、输出 $[B, T, d]$”的形状守恒契约，主干网络才能把 $L = 30$ 层结构完全相同（但各自拥有独立训练参数）的 Transformer 层像接力赛一样直接串联起来：

$$
\mathbf{H}^{(0)} \xrightarrow{\text{Layer } 0} \mathbf{H}^{(1)} \xrightarrow{\text{Layer } 1} \cdots \xrightarrow{\text{Layer } 29} \mathbf{H}^{(30)} \xrightarrow{\text{Final RMSNorm}} \mathbf{H}_{\text{final}}: [B, T, d]
$$

- **浅层**（如第 $0 \dots 5$ 层）：主要整合局部相邻词的词法与短语搭配；
- **中层**（如第 $6 \dots 20$ 层）：梳理跨句子的代词指代、语法角色与逻辑结构；
- **深层**（如第 $21 \dots 29$ 层）：将上下文信息逐步收敛到“当前位置接下来应该输出什么词”的预测方向上。

走出第 29 层后，张量经过全模型第 272 个权重 `model.norm.weight`（形状 `[576]`）做最后一次全局 RMSNorm 幅度校准，得到最终隐藏状态 $\mathbf{H}_{\text{final}}: [B, T, 576]$。

---

### 09. 输出投影（LM Head）与权重绑定

现在，我们手头有了每个位置的 576 维最终特征向量 $\mathbf{H}_{\text{final}}: [B, T, 576]$。怎样把这 576 个浮点数翻译回词表里 $V = 49,152$ 个候选词各自的预测得分？

还记得阶段 02 入口处那张形状为 `[49152, 576]` 的词嵌入表 $\mathbf{W}_E$ 吗？

- **在阶段 02 入口处**：我们拿整数行号去表里查出对应的 `576` 维词向量；
- **在阶段 09 出口处**：我们正好把这个过程倒过来——把这张表转置为 **`[576, 49152]`**（充当**语言模型输出头 LM Head**，$\mathbf{W}_{\text{head}}^\top$），让 $\mathbf{H}_{\text{final}}: [B, T, 576]$ 与这 `49,152` 列候选词向量逐一做点积！

$$
\text{logits}_{\text{all}} = \mathbf{H}_{\text{final}} \mathbf{W}_{\text{head}}^\top, \qquad \text{Shape: } [B, T, 576] \times [576, 49152] \longrightarrow [B, T, 49152]
$$

从向量点积的几何意义来看，这一步再直观不过了：**最终隐藏状态 $\mathbf{H}_{\text{final}}$ 与词表里第 $k$ 号候选词的向量方向越接近，点积算出来的得分（称为未归一化对数概率 Logits）就越高！**

> **首尾词表权重绑定**（Weight Tying）：
> - **3B 及以下轻量模型**（`tie_word_embeddings: true`）：如 `SmolLM2-135M`、`Qwen2.5-0.5B`、`Llama-3.2-1B`。由于词表矩阵占比极高（本模型单张表即占总参数的 21.05%），出口处的 $\mathbf{W}_{\text{head}}$ 在物理内存中直接复用了阶段 02 入口处的 `model.embed_tokens.weight`，因此在 `.safetensors` 中不存在独立的 `lm_head.weight` 键；
> - **7B 及以上中大型模型**（`tie_word_embeddings: false`）：参数预算充裕，为避免同一个矩阵在训练时同时承受入口特征提取与出口分类器两股方向的梯度拉扯，入口 `embed_tokens.weight` 与出口 `lm_head.weight` 解绑为两张独立的 `[V, d]` 矩阵。

---

### 10. 选出下一个 Token

看着形状为 $[B, T, V]$（如 `[1, 3, 49152]`）的 $\text{logits}_{\text{all}}$，初学者常会问：**我们明明只想要预测整句话后面的那 1 个新词，为什么模型一口气算出了全部 $T = 3$ 行的 49,152 维得分？前两行是干嘛的？**

这正是阶段 05 因果掩码留下的结构印记（以输入 `"我 喜欢 写"` 为例）：

- **第 $0$ 行**（$t=0$，此时只看到 `"我"`）：预测的是 `"我"` 后面该接什么词；
- **第 $1$ 行**（$t=1$，此时看到 `"我 喜欢"`）：预测的是 `"我 喜欢"` 后面该接什么词；
- **最后一行**（$t=T-1=2$，此时看全了整句 `"我 喜欢 写"`）：预测的才是整句话后面的**下一个新词**！

在**训练阶段**，前 $T-1$ 行的得分至关重要——训练框架会同时拿全部 $T$ 行的预测得分与真实语料中的下一个词比对算损失，用一次前向传播同时训练 $T$ 个预测任务。

但在我们今天运行的**推理生成阶段**，前 $T-1$ 个词本来就是用户已经敲进去的已知文本，无需重猜。因此程序直接丢弃前 $T-1$ 行，只切出**序列最后一个位置**（下标 $T-1$）的那一行 `[B, V]` 得分，并通过 $\arg\max$（贪心解码）挑出得分最高的那一列下标：

$$
\text{logits}_{\text{last}} = \text{logits}_{\text{all}}[:, T-1, :] \in \mathbb{R}^{[B, V]}, \qquad \text{next\_id} = \arg\max(\text{logits}_{\text{last}})
$$

最后拿这个整数 `next_id` 反查 `tokenizer.json` 词表，就还原出了模型生成的下一个词（如 `"代码"`）。

---

### 11. 自回归生成循环（Autoregressive Loop）

既然跑完一整趟阶段 01 — 10 只能吐出 **1 个新 Token**，大模型是怎么写出一整段回答的？

答案是**自回归生成循环**（Autoregressive Generation）——把上一步刚生成的 Token 拼回输入序列的末尾，作为下一步的输入再跑一轮前向计算：

1. **第 1 轮**：输入 `[1, 3, 4]`（`"我 喜欢 写"`，长 $3$） $\xrightarrow{\text{前向计算}}$ 生成 Token ID `5`（`"代码"`）；
2. **第 2 轮**：将 `5` 拼在末尾，输入变为 `[1, 3, 4, 5]`（`"我 喜欢 写 代码"`，长 $4$） $\xrightarrow{\text{前向计算}}$ 生成下一个 Token；
3. **终止条件**：如此循环往复，直到模型在某一轮挑中了代表回答结束的特殊控制词元 `<|im_end|>`（EOS Token，ID 为 `2`），生成循环正式结束。

---

## 八、 教学模型与现代开源架构对比

现代开源架构（Llama / Qwen / SmolLM2）在保持 `[B, T] -> [B, T, d] -> [B, T, V]` 主干契约不变的前提下，相对于早期 GPT-2 教学模型完成了六项标准化演进：

| 演进维度 | 早期教学模型 (GPT-2) | 现代开源架构 (Llama / SmolLM2) | 权重与激活形状影响 |
|---|---|---|---|
| **1. 归一化** | Post-LayerNorm（含 `bias`） | **Pre-RMSNorm**（无 `bias`） | 移除偏置 `[d]`，形状守恒为 `[B, T, d]` |
| **2. 位置编码** | 绝对位置编码 $\mathbf{W}_{\text{pos}}: [T_{\max}, d]$ | **旋转位置编码 (RoPE)** | 移除 $\mathbf{W}_{\text{pos}}$，在 `[B, h, T, d_h]` 上原地旋转 |
| **3. 注意力头** | 标准多头 MHA ($h_q = h_{kv}$) | **分组查询 GQA** ($h_q = g \cdot h_{kv}$) | $\mathbf{W}_K, \mathbf{W}_V$ 缩小为 `[h_kv * d_h, d]` |
| **4. 前馈网络** | 双矩阵 MLP (`up` + GELU + `down`) | **三矩阵 SwiGLU** (`gate` $\odot$ `up` + `down`) | 权重增至 3 张，两路 `[B, T, d_ff]` 相乘汇合 |
| **5. 线性层偏置** | 全部保留 `.bias: [d_out]` | **普遍移除 (`bias=False`)** | 移除一维偏置向量 |
| **6. 首尾词表** | 强制绑定 | **3B 以下绑定，7B 以上解绑** | 3B 以下无 `lm_head.weight` 独立键 |

横向对比四款主流轻量级开源模型的架构尺寸：

| 指标 | `SmolLM2-135M` | `Qwen2.5-0.5B` | `Llama-3.2-1B` | `Gemma-3-1B` |
|---|---|---|---|---|
| **隐藏维度 $d$ / 层数 $L$** | $576$ / $30$ 层 | $896$ / $24$ 层 | $2048$ / $16$ 层 | $1152$ / $26$ 层 |
| **头配置 ($h_q / h_{kv} / d_h$)** | $9 / 3 / 64$ | $14 / 2 / 64$ | $32 / 8 / 64$ | $4 / 1 / 256$ |
| **MLP 宽度 $d_{ff}$** | $1536$ | $4864$ | $8192$ | $6912$ (GeGLU) |
| **词表大小 $V$** | $49,152$（绑定） | $151,936$（绑定） | $128,256$（绑定） | $262,144$（绑定） |
| **张量总数 / 体积** | **272 个** / 269 MB | **290 个** / 988 MB | **146 个** / 2.47 GB | **340 个** / 2.0 GB |

- **`SmolLM2-135M`**：零偏置标准 Llama 架构，CPU 仅需 0.3 秒即可跑完，是最佳白盒追踪基准；
- **`Qwen2.5-0.5B`**：与 `SmolLM2` 结构一致，仅在每层 `q/k/v_proj` 保留了一维 `.bias` 偏置向量（每层 12 个张量）；
- **`Llama-3.2-1B`**：启用 `llama3` 分段波长 RoPE 缩放以支持长上下文；
- **`Gemma-3-1B`**：查询头总宽 $4 \times 256 = 1024 \neq 1152$，包含 QK-Norm 与 5:1 混合滑动窗口注意力（每层 13 个张量）。

---

## 九、 真机前向追踪示例

现在我们打开 `crates/llm-tracer` 的最后两个源码文件：`src/ops.rs` 与 `src/model.rs`。

### 1. 基础算子实现（src/ops.rs）

跑通整座 Llama 架构大模型，底层只需实现 **5 个数学算子函数**（共 104 行）：

```rust
// crates/llm-tracer/src/ops.rs（5 个核心算子）

/// 1. 无偏置线性投影：[B, T, d_in] -> [B*T, d_in] x [d_in, d_out] -> [B, T, d_out]
pub fn linear_no_bias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (b, t, d_in) = x.dims3()?;
    let (d_out, _) = weight.dims2()?;
    x.reshape((b * t, d_in))?.matmul(&weight.t()?)?.reshape((b, t, d_out))
}

/// 2. RMSNorm 均方根归一化：x / sqrt(mean(x^2) + eps) * weight
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    candle_nn::ops::rms_norm(x, weight, eps as f32)
}

/// 3. 预计算 RoPE 余弦与正弦查找表（形状均为 [max_seq_len, d_h / 2]）
pub fn precompute_rope_tables(max_seq_len: usize, d_h: usize, theta: f32, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = d_h / 2;
    let inv_freq: Vec<f32> = (0..half).map(|i| 1.0 / theta.powf((2 * i) as f32 / d_h as f32)).collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
    let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
    let t = Tensor::from_vec(positions, (max_seq_len, 1), dev)?;
    let freqs = t.matmul(&inv_freq)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

/// 4. GQA 键值头复制扩展：[B, h_kv, T, d_h] -> [B, h_q, T, d_h]
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 { return Ok(x.clone()); }
    let (b, h_kv, seq_len, d_h) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, h_kv, n_rep, seq_len, d_h))?.reshape((b, h_kv * n_rep, seq_len, d_h))
}

/// 5. 上三角因果掩码：将 j > i + offset 的未来位置置为 -inf
pub fn apply_causal_mask(scores: &Tensor, offset: usize) -> Result<Tensor> {
    let (b, h, q_len, kv_len) = scores.dims4()?;
    if q_len == 1 { return Ok(scores.clone()); }
    let mut mask_vec = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in (i + offset + 1)..kv_len {
            mask_vec[i * kv_len + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(mask_vec, (1, 1, q_len, kv_len), scores.device())?.broadcast_as((b, h, q_len, kv_len))?;
    scores.broadcast_add(&mask)
}
```

### 2. 11 阶段前向主循环（src/model.rs）

`src/model.rs` 中的 `forward_step` 函数将上述 5 个算子串联为阶段 01 — 10 的完整前向计算：

```rust
// crates/llm-tracer/src/model.rs —— forward_step 核心实现
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
    let (b, seq_len) = input_ids.dims2()?;
    let (d, h_q, h_kv, d_h) = (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim());
    let n_rep = h_q / h_kv;

    // 阶段 01 & 02：Embedding 按行查表 [B, T] -> [B, T, 576]
    let w_embed = get_tensor(weights, "model.embed_tokens.weight")?;
    let mut hidden = w_embed.index_select(&input_ids.flatten_all()?, 0)?.reshape((b, seq_len, d))?;

    let cos = cos_all.narrow(0, offset, seq_len)?;
    let sin = sin_all.narrow(0, offset, seq_len)?;

    // 阶段 03 — 08：循环穿过 30 层 Transformer
    for (l, kv_slot) in kv_caches.iter_mut().enumerate().take(cfg.num_hidden_layers) {
        let p = format!("model.layers.{l}");

        // 自注意力子层：Pre-RMSNorm -> QKV 投影与拆头 -> RoPE 旋转 -> KV Cache 拼接 -> GQA 注意力 -> W_O 投影 + 残差
        let x_norm = rms_norm(&hidden, get_tensor(weights, &format!("{p}.input_layernorm.weight"))?, cfg.rms_norm_eps)?;
        let q = linear_no_bias(&x_norm, get_tensor(weights, &format!("{p}.self_attn.q_proj.weight"))?)?
            .reshape((b, seq_len, h_q, d_h))?.transpose(1, 2)?.contiguous()?;
        let k = linear_no_bias(&x_norm, get_tensor(weights, &format!("{p}.self_attn.k_proj.weight"))?)?
            .reshape((b, seq_len, h_kv, d_h))?.transpose(1, 2)?.contiguous()?;
        let v = linear_no_bias(&x_norm, get_tensor(weights, &format!("{p}.self_attn.v_proj.weight"))?)?
            .reshape((b, seq_len, h_kv, d_h))?.transpose(1, 2)?.contiguous()?;

        let q_rope = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_rope = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;

        let (k_cache, v_cache) = match kv_slot.take() {
            Some((prev_k, prev_v)) => (Tensor::cat(&[&prev_k, &k_rope], 2)?, Tensor::cat(&[&prev_v, &v], 2)?),
            None => (k_rope, v),
        };
        *kv_slot = Some((k_cache.clone(), v_cache.clone()));

        let k_full = repeat_kv(&k_cache, n_rep)?;
        let v_full = repeat_kv(&v_cache, n_rep)?;
        let attn_scores = (q_rope.matmul(&k_full.t()?)? * (1.0 / (d_h as f64).sqrt()))?;
        let attn_probs = candle_nn::ops::softmax_last_dim(&apply_causal_mask(&attn_scores, offset)?)?;
        let o_concat = attn_probs.matmul(&v_full.contiguous()?)?.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, d))?;
        let h_mid = (&hidden + &linear_no_bias(&o_concat, get_tensor(weights, &format!("{p}.self_attn.o_proj.weight"))?)?)?;

        // SwiGLU MLP 子层：Pre-RMSNorm -> SiLU(gate) * up -> down + 残差
        let h_norm = rms_norm(&h_mid, get_tensor(weights, &format!("{p}.post_attention_layernorm.weight"))?, cfg.rms_norm_eps)?;
        let gate = candle_nn::ops::silu(&linear_no_bias(&h_norm, get_tensor(weights, &format!("{p}.mlp.gate_proj.weight"))?)?)?;
        let up = linear_no_bias(&h_norm, get_tensor(weights, &format!("{p}.mlp.up_proj.weight"))?)?;
        let mlp_out = linear_no_bias(&(&gate * &up)?, get_tensor(weights, &format!("{p}.mlp.down_proj.weight"))?)?;
        hidden = (&h_mid + &mlp_out)?;
    }

    // 阶段 08 — 10：最终归一化 -> LM Head 词表投影 -> 取末位 argmax
    let h_final = rms_norm(&hidden, get_tensor(weights, "model.norm.weight")?, cfg.rms_norm_eps)?;
    let logits = linear_no_bias(&h_final, w_embed)?;
    logits.i((0, seq_len - 1))?.argmax(0)?.to_scalar::<u32>()
}
```

### 3. 运行 --mode real 观察 Step 1 日志

在终端执行真机追踪：

```bash
cargo run -- --mode real
```

终端首先打印 **Step 1**（Prefill 预填充阶段，输入 10 个 Prompt Token） 的追踪日志：

```text
[01.Tokenizer ] Raw prompt: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
[01.Tokenizer ] Tokens (10): ["<|im_start|>", "user", "Ċ", "Hello", "<|im_end|>", "Ċ", "<|im_start|>", "ass", "istant", "Ċ"]
[01.Tokenizer ] Token IDs : [1, 4093, 198, 19556, 2, 198, 1, 520, 9531, 198]

==========================================================================================
=== STEP 1: PREFILL (T = 10 ChatML prompt tokens) (offset = 0) ===
==========================================================================================
[01.Tokens     ] input_ids                            | ACTIVATION | shape: [1, 10]              | [+1.0000, +4093.0000, +198.0000, +19556.0000 ...]
[02.Embedding  ] model.embed_tokens.weight            | WEIGHT     | shape: [49152, 576]         | [-0.1060, +0.0342, +0.0476, -0.0239 ...]
[02.Embedding  ] X_0 (embedded hidden state)          | ACTIVATION | shape: [1, 10, 576]         | [-0.0255, -0.0192, -0.0396, +0.0187 ...]
[03.PreRMSNorm ] model.layers.0.input_layernorm.weight | WEIGHT     | shape: [576]                | [+0.0140, +0.0238, -0.0198, -0.0303 ...]
[03.PreRMSNorm ] X_norm (Layer 0)                     | ACTIVATION | shape: [1, 10, 576]         | [-0.0026, -0.0033, +0.0057, -0.0041 ...]
[04.AttnProj   ] model.layers.0.self_attn.q_proj.weight | WEIGHT     | shape: [576, 576]           | [-0.0752, +0.1621, -0.0991, -0.7305 ...]
[04.AttnProj   ] model.layers.0.self_attn.k_proj.weight | WEIGHT     | shape: [192, 576]           | [-0.1162, +0.2695, -0.4180, -0.2490 ...]
[04.AttnProj   ] model.layers.0.self_attn.v_proj.weight | WEIGHT     | shape: [192, 576]           | [+0.1055, +0.0757, -0.0903, +0.0248 ...]
[04.AttnProj   ] Q_lin                                | ACTIVATION | shape: [1, 10, 576]         | [-0.4388, -0.1722, +0.0118, -1.3227 ...]
[04.AttnProj   ] K_lin                                | ACTIVATION | shape: [1, 10, 192]         | [-3.1820, +0.1983, -0.5812, -1.0393 ...]
[04.AttnProj   ] V_lin                                | ACTIVATION | shape: [1, 10, 192]         | [-0.0210, +0.0031, -0.0384, -0.0375 ...]
[03.RoPE       ] Q_rope (rotated queries)             | ACTIVATION | shape: [1, 9, 10, 64]       | [-0.4388, -0.1722, +0.0118, -1.3227 ...]
[03.RoPE       ] K_rope (rotated keys)                | ACTIVATION | shape: [1, 3, 10, 64]       | [-3.1820, +0.1983, -0.5812, -1.0393 ...]
[05.KVCache    ] Layer 0 K_cache                      | KV_CACHE   | shape: [1, 3, 10, 64]       | [-3.1820, +0.1983, -0.5812, -1.0393 ...]
[05.KVCache    ] Layer 0 V_cache                      | KV_CACHE   | shape: [1, 3, 10, 64]       | [-0.0210, +0.0031, -0.0384, -0.0375 ...]
[05.Attention  ] K_full (after repeat_kv)             | ACTIVATION | shape: [1, 9, 10, 64]       | [-3.1820, +0.1983, -0.5812, -1.0393 ...]
[05.Attention  ] Scores (QK^T / sqrt(d_h))            | ACTIVATION | shape: [1, 9, 10, 10]       | [+1.7113, -inf, -inf, -inf ...]
[05.Attention  ] Probs (Softmax)                      | ACTIVATION | shape: [1, 9, 10, 10]       | [+1.0000, +0.0000, +0.0000, +0.0000 ...]
[05.Attention  ] Context (Probs * V)                  | ACTIVATION | shape: [1, 9, 10, 64]       | [-0.0210, +0.0031, -0.0384, -0.0375 ...]
[06.MergeOut   ] O_concat                             | ACTIVATION | shape: [1, 10, 576]         | [-0.0210, +0.0031, -0.0384, -0.0375 ...]
[06.MergeOut   ] model.layers.0.self_attn.o_proj.weight | WEIGHT     | shape: [576, 576]           | [+0.0835, +0.0801, +0.0552, +0.0189 ...]
[06.MergeOut   ] H_mid (after Attn residual)          | ACTIVATION | shape: [1, 10, 576]         | [+0.1128, +0.2753, -0.2110, -0.0316 ...]
[07.SwiGLU_MLP ] model.layers.0.mlp.gate_proj.weight  | WEIGHT     | shape: [1536, 576]          | [-0.0145, +0.1826, +0.1377, -0.2314 ...]
[07.SwiGLU_MLP ] Gate = SiLU(H_norm * W_gate)         | ACTIVATION | shape: [1, 10, 1536]        | [+0.1604, +1.2164, +0.1433, +0.6469 ...]
[07.SwiGLU_MLP ] Up = H_norm * W_up                   | ACTIVATION | shape: [1, 10, 1536]        | [-0.2131, -0.7661, -0.6609, -0.3793 ...]
[07.SwiGLU_MLP ] model.layers.0.mlp.down_proj.weight  | WEIGHT     | shape: [576, 1536]          | [-0.3965, +0.2930, +0.0286, -0.4023 ...]
[07.SwiGLU_MLP ] X_1 (Layer 0 output)                 | ACTIVATION | shape: [1, 10, 576]         | [+5.3622, -0.2572, -11.1267, -1.1553 ...]
[08.RepeatL30  ] X_30 (after 30 layers + norm)        | ACTIVATION | shape: [1, 10, 576]         | [-0.7521, -0.1531, +0.1131, +0.7997 ...]
[09.LM_Head    ] lm_head.weight (tied W_E)            | WEIGHT     | shape: [49152, 576]         | [-0.1060, +0.0342, +0.0476, -0.0239 ...]
[09.LM_Head    ] Logits [B, T, V]                     | ACTIVATION | shape: [1, 10, 49152]       | [+14.3638, +5.7890, +8.8255, +5.2430 ...]
[10.SelectNext ] last_logits [V]                      | ACTIVATION | shape: [49152]              | [+21.0763, +22.4347, +24.7866, +20.4223 ...]
[10.SelectNext ] Greedy argmax next_token_id = 19556
[10.Decoded    ] Step 1 generated token 19556 -> "Hello"
```

日志清晰印证了前文的三条核心规律：

- **维度铁律**：所有 `WEIGHT` 行均不含批大小 `1` 与序列长 `10`；所有 `ACTIVATION` 行均携带 `1` 与 `10`。
- **因果掩码效应**：`Scores` 第 0 行后三个元素全为 `-inf`，`Probs` 变为 `[+1.0000, +0.0000, ...]`，使得 `Context` 的前 4 个浮点数 `[-0.0210, +0.0031, -0.0384, -0.0375]` 与 `V_lin` 完全相同。
- **首词解码**：30 层输出 `X_30` 乘转置词表后，末位 `last_logits` 在第 `19556` 列取得最高分，反查 `tokenizer.json` 解码出首个回答词 **`"Hello"`**。

---

## 十、 Prefill、Decode 与 KV Cache

### 1. 如果没有缓存，自回归循环会遇到什么算力灾难？

在 Step 1 中，我们输入了 10 个 Token（`[1, 10]`），跑完 30 层网络后生成了第 11 个 Token（`19556`, `"Hello"`）。

到了 Step 2，我们要继续预测 `"Hello"` 后面的标点符号。如果按照最朴素的做法，把全部 11 个 Token（`[1, 11]`）重新从头喂进模型：

- 前 10 个 Token 的 Embedding 查表要重做一遍；
- 前 10 个 Token 在 30 层里的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 线性投影与 MLP 计算要全部重算一遍！
- 当生成到第 1,000 个词时，每吐出 1 个新字，都要把前面 999 个词在 30 层里的矩阵乘法完完全全重算一遍！

### 2. 为什么前 10 个词的 K 和 V 永远不会变？（因果掩码的馈赠）

仔细想一想：当第 11 个词加到句子末尾时，前 10 个词在每一层算出来的向量真的会改变吗？

**绝对不会改变！** 回想阶段 05 的**因果掩码**：因为右上角未来位置全被抹成了 $-\infty$，前 10 个词在计算注意力时**根本看不见第 11 个词**。因此，无论后面追加多少个新词，前 10 个位置在每一层算出的键向量 $\mathbf{K}_{1..10}$ 和值向量 $\mathbf{V}_{1..10}$，连小数点后最后一位都分毫不差。

那么，在计算第 11 个词（下标 $t=10$）的注意力输出时，到底需要用到前 10 个词的什么数据？

只需用到前 10 个词在各层的 **键向量 $\mathbf{K}_{1..10}$** 和 **值向量 $\mathbf{V}_{1..10}$**！

```text
Step 2 (Decode) 只输入第 11 个新词 (T_new = 1)：
  当前新词自己的查询向量：q_11       形状 [1, 9,  1, 64]
  当前新词算出的新键向量：k_11       形状 [1, 3,  1, 64] ──┐
  Step 1 已存好的历史键：K_{1..10}   形状 [1, 3, 10, 64] ──┴─► 沿序列维 (dim=2) 拼接为 K_{1..11}: [1, 3, 11, 64]
```

这就是 **KV Cache**（键值缓存）的全部原理：

- **缓存什么**：在显存中把每一层已经算过的历史位置的 $\mathbf{K}_{\text{rope}}$ 和 $\mathbf{V}$ 张量保存下来（按未广播的 $h_{kv} = 3$ 个键值头存储以省显存）；每当新词进来，只需算出这 **1 个新词**自己的 $\mathbf{k}_{\text{new}}, \mathbf{v}_{\text{new}}$（形状 `[1, 3, 1, 64]`），调用 `Tensor::cat` 拼在历史缓存末尾，就得到了完整的 `[1, 3, 11, 64]`！
- **为什么不缓存 $\mathbf{Q}$**：因为前 10 个位置的历史查询向量 $\mathbf{Q}_{1..10}$ 只用于预测第 $1 \dots 10$ 个位置各自的下一个词，那些词早已生成完毕，后续步骤永远不会再回头使用旧的 $\mathbf{Q}$。

### 3. Step 2（Decode 阶段）真机日志

在 `--mode real` 输出的下半部分，模型仅喂入刚生成的单Token `[19556]`（形状 `[1, 1]`，`offset = 10`）：

```text
==========================================================================================
=== STEP 2: AUTOREGRESSIVE DECODE (T_new = 1, KV Cache 10 -> 11) (offset = 10) ===
==========================================================================================
[01.Tokens     ] input_ids                            | ACTIVATION | shape: [1, 1]               | [+19556.0000 ...]
[02.Embedding  ] X_0 (embedded hidden state)          | ACTIVATION | shape: [1, 1, 576]          | [-0.0498, +0.0398, +0.0012, +0.0859 ...]
[03.RoPE       ] Q_rope (rotated queries)             | ACTIVATION | shape: [1, 9, 1, 64]        | [+0.4956, +0.2792, +0.2399, +0.7698 ...]
[03.RoPE       ] K_rope (rotated keys)                | ACTIVATION | shape: [1, 3, 1, 64]        | [+1.8024, +0.0890, +0.5022, +1.0591 ...]
[05.KVCache    ] Layer 0 K_cache                      | KV_CACHE   | shape: [1, 3, 11, 64]       | [-3.1820, +0.1983, -0.5812, -1.0393 ...]
[05.Attention  ] K_full (after repeat_kv)             | ACTIVATION | shape: [1, 9, 11, 64]       | [-3.1820, +0.1983, -0.5812, -1.0393 ...]
[05.Attention  ] Scores (QK^T / sqrt(d_h))            | ACTIVATION | shape: [1, 9, 1, 11]        | [+0.5725, +0.5533, +1.0500, +0.6461 ...]
[05.Attention  ] Probs (Softmax)                      | ACTIVATION | shape: [1, 9, 1, 11]        | [+0.0614, +0.0602, +0.0990, +0.0661 ...]
[06.MergeOut   ] O_concat                             | ACTIVATION | shape: [1, 1, 576]          | [-0.0019, +0.0136, -0.0206, -0.0091 ...]
[08.RepeatL30  ] X_30 (after 30 layers + norm)        | ACTIVATION | shape: [1, 1, 576]          | [-1.2330, +0.7028, +0.8036, +1.5715 ...]
[09.LM_Head    ] Logits [B, T, V]                     | ACTIVATION | shape: [1, 1, 49152]        | [+7.5679, +15.7363, +21.9471, +10.1808 ...]
[10.SelectNext ] Greedy argmax next_token_id = 17
[10.Decoded    ] Step 2 generated token 17 -> "!"
[11.Summary    ] First 2 generated tokens decoded: "Hello!"
```

Step 2 选出最高分 Token ID `17`（`"!"`），与 Step 1 的 `"Hello"` 拼接输出 **`"Hello!"`**。

### 4. Prefill 与 Decode 的形状对比

开启 KV Cache 后，推理过程自然分为两个阶段：

| 追踪节点 | Prefill 阶段（$T=10$） | Decode 阶段（$T_{\text{new}}=1$） |
|---|---|---|
| **输入 `input_ids`** | `[1, 10]` | **`[1, 1]`** |
| **层间隐藏状态 $\mathbf{H}$** | `[1, 10, 576]` | **`[1, 1, 576]`** |
| **新投影 $\mathbf{Q}_{\text{new}}$** | `[1, 9, 10, 64]` | **`[1, 9, 1, 64]`** |
| **新投影 $\mathbf{K}_{\text{new}}, \mathbf{V}_{\text{new}}$** | `[1, 3, 10, 64]` | **`[1, 3, 1, 64]`** |
| **拼接后 $\mathbf{K}_{\text{all}}, \mathbf{V}_{\text{all}}$** | `[1, 3, 10, 64]` | **`[1, 3, 11, 64]`** |
| **注意力分数 $\mathbf{Q}_{\text{new}}\mathbf{K}_{\text{all}}^\top$** | **`[1, 9, 10, 10]`**（二维方阵） | **`[1, 9, 1, 11]`**（单行向量） |
| **MLP 升维 / Logits** | `[1, 10, 1536]` $\to$ `[1, 10, 49152]` | **`[1, 1, 1536]`** $\to$ **`[1, 1, 49152]`** |

- **Prefill**（计算受限 Compute-Bound）：一次性并行处理整句 Prompt，做矩阵乘矩阵运算并填满 KV Cache；
- **Decode**（访存带宽受限 Memory-Bound）：每步仅输入 1 个新词，线性层与 MLP 的激活退化为单行向量 `[1, 1, d]`，主要耗时花在从显存搬运庞大的权重与历史 KV Cache 上。

---

## 十一、 前沿对标：DeepSeek-V4.1-Flash

掌握了 `SmolLM2-135M` 的 11 阶段流转后，我们来看工业界前沿旗舰模型 **`DeepSeek-V4.1-Flash-1017`**（5520 亿总参数、支持 100 万 Token 上下文，词表 $V = 129,280$、隐藏维度 $d = 5120$、层数 $L = 40$）。

它的宏观主干依然是 `[B, T] -> [B, T, 5120] -> [B, T, 129280]`，但在四个核心算力与显存瓶颈处实施了外科手术式演进：

| 演进部位 | `SmolLM2-135M`（今日基准） | `DeepSeek-V4.1-Flash-1017`（前沿标杆） | 解决的核心瓶颈 |
|---|---|---|---|
| **1. MLP 与知识存储** | 稠密 SwiGLU（100% 激活） | **细粒度 MoE（384 选 6）+ 196B Engram 记忆** | 总容量 552B，单步仅百亿级算力 |
| **2. 层堆叠与读写分工** | 30 层均质纯 Decoder | **20+20 层非对称 CED**（Prefill 8B / Decode 16B） | 长文预填充算力直接砍半 |
| **3. 注意力与 KV 缓存** | 标准 GQA（FP16/BF16 缓存） | **MLA 低秩压缩 + CSA2 稀疏 + FP4 量化** | 全局 KV 压至 **890 Bytes/词** |
| **4. 自回归解码步进** | 单步串行吐 1 词 | **DSpark 投机解码 + `1017` RL 思维链精简** | 一次验真多词，减少总输出步数 |

### 1. 阶段 07 演进：MoE 专家路由与 Engram 条件记忆

在稠密模型中，每个 Token 在阶段 07 都要乘完整的 MLP 矩阵。`DeepSeek-V4.1-Flash` 将单层 MLP 拆分为 **1 个共享专家 + 384 个路由专家**，每个词仅通过轻量路由矩阵唤醒得分最高的 **6 个专家**参与计算；同时引入 **1960 亿参数的 Engram 条件记忆模块**，通过稀疏查表直接提取静态事实知识，省去大量矩阵乘法。

### 2. 阶段 08 演进：CED 非对称因果编码-解码器

针对智能体与代码场景“长文输入（Prefill 重）、短补丁输出（Decode 轻）”的特征，模型将 40 层划分为 **20 层因果编码器 + 20 层解码器**。Prefill 阶段每个词仅激活 **8B 参数**快速消化长上下文，并在第 20 层出口直接生成全局 KV Cache；Decode 阶段则激活 **16B 参数**保障代码与推理生成质量。

### 3. 第十节演进：MLA + CSA2 + FP4 极低显存 KV Cache

为支撑 100 万 Token 上下文，模型结合了三项缓存压缩技术：**MLA**（多头潜在注意力） 将键值张量压入低维潜在向量（单独分离 `qk_rope_head_dim = 64` 施加旋转编码）；**CSA2**（第二代压缩稀疏注意力） 按 `Full / Reindex / Reuse` 跨层复用稀疏 Top-K 位置索引；配合 **FP4（E2M1）缓存量化**，将单 Token 的全模型 KV Cache 压缩至 **890 字节**。

### 4. 阶段 11 演进：DSpark 投机解码与 1017 强化学习精简

为打破阶段 11 单步串行吐词的延迟瓶颈，模型引入 **DSpark 半自回归投机解码**，单次预测多个后续 Token 并并行验真；同时 `1017` 版本通过自反馈强化学习（RL）精简了冗余的中间思维链，以更少的解码步数完成工程交付。

---

## 十二、 结语与机械可解释性边界

至此，看到任意一个张量形状，你都可以立即定位它在计算图中的坐标：

- 看到 `[1, 3, 10, 64]`：带有 $B=1, T=10$，属于**动态中间激活**（或 KV Cache），代表 3 个 GQA 键值头、单头 64 维；
- 看到 `[1536, 576]`：不含 $B$ 与 $T$，属于 `.safetensors` 中的**静态参数矩阵**（SwiGLU 的 `gate_proj` 或 `up_proj` 权重）。

但在掌握张量流转之后，我们还需明确一条科学认知边界：**看清每一个张量的维度与数值来路，不等于直接读懂了模型的内部语义。**

在**机械可解释性**（Mechanistic Interpretability）研究中，三项基础定理揭示了为何不能靠肉眼将单个浮点数或注意力分数直接等同于人类概念：

### 1. 残差流的“无特权基（No Privileged Basis）”

如果向量空间的各个坐标轴天然具有特殊地位，使模型倾向于将独立特征对齐到单个坐标轴下标上，这组坐标基就称为**特权基**（Privileged Basis）。

然而，残差流 `[B, T, d]` 是各层组件仅通过线性矩阵乘法读写、通过线性加法叠加的纯线性通信总线（Elhage et al., 2021）。由前置篇证明的**正交基变换不变性** $(\mathbf{x}\mathbf{R})(\mathbf{R}^\top\mathbf{W}) = \mathbf{x}\mathbf{W}$，若将残差流乘任意正交旋转矩阵 $\mathbf{R}$、同时将读写权重乘 $\mathbf{R}^\top$，残差流在单个坐标下标上的数值会全部改变，但全模型输出完全等价。

因此，**残差流不存在特权基**——第 $k$ 个浮点数下标只是人为坐标系下的任意投影，模型是以任意旋转方向上的**低维线性子空间**（Linear Subspace）为单位编码信息的。

### 2. 多义性（Polysemanticity）与叠加态假设（Superposition）

即便在带有逐元素 $\operatorname{SiLU}$ 非线性激活（从而打破正交旋转对称性、拥有特权基）的 MLP 隐藏层 $[B, T, d_{ff}]$ 中，单个神经元也极少只对应单一概念（**单义性 Monosemanticity**），而是普遍表现出**多义性**（Polysemanticity）——同一个神经元会在完全无关的语境下同时强烈激活。

Anthropic（2022）提出的**叠加态假设**（Superposition Hypothesis）指出：现实世界中的稀疏语义特征数量远大于隐藏维度 $d$。模型利用前置篇介绍的 **Johnson-Lindenstrauss 引理**，在 $d$ 维空间中选出多达 $\exp(c \cdot \epsilon^2 d)$ 个**近似正交**（Almost Orthogonal）的单位方向，将成千上万个稀疏特征线性叠加在同一个 $d$ 维空间中。单个坐标下标上的数值，只是众多高维特征方向在该坐标轴上的重叠投影。

### 3. QK 寻址与 OV 搬运分离及注意力汇聚点（Attention Sink）

形状为 `[B, h, T, T]` 的注意力权重 $\alpha_{i,j}$ 高，并不等于位置 $j$ 对位置 $i$ 的输出贡献大。将阶段 04 — 06 展开，单个注意力头对残差流的写入增量可严格分解为两条独立电路：

- **QK 寻址电路**（$\mathbf{W}_Q \mathbf{W}_K^\top$）：决定注意力分数与 Softmax 权重 $\alpha_{i,j}$，仅负责计算**去哪个位置寻址**；
- **OV 搬运电路**（$\mathbf{W}_V \mathbf{W}_O$）：决定从位置 $j$ 提取并写回残差流的实际向量 $\Delta\mathbf{x}_i = \sum_j \alpha_{i,j} (\mathbf{x}_j \mathbf{W}_V \mathbf{W}_O)$，负责决定**搬运什么内容及幅度**。

由于 Softmax 强制每行概率和恒为 $1.0$，当某个注意力头在当前语境下无需触发时，QK 电路常将接近 `1.0` 的极高权重倾倒在句首控制符 `<|im_start|>` 上（称为 **Attention Sink 注意力汇聚点**，Xiao et al., 2023），同时 OV 电路令该词元的映射范数 $\|\mathbf{x}_{\text{sink}} \mathbf{W}_V \mathbf{W}_O\| \approx 0$，从而用高注意力权重实现了一次**空操作**（No-Op）。

### 4. 机械可解释性的四级科学取证工具

正因如此，研究者在考证模型内部机制时，依赖以下四级递进的受控实验工具：

| 取证方法 | 核心操作 | 证据层级 | 适用问题 |
|---|---|---|---|
| **1. 对数几率透镜 (Logit Lens)** | 中间层 $\mathbf{H}^{(\ell)}$ 提前乘 $\mathbf{W}_{\text{head}}^\top$ | 直投观测级 | 观测候选词分布在 30 层间的收敛过程 |
| **2. 线性探针 (Linear Probing)** | 在冻结激活上训练外部线性分类器 | 相关性观测级 | 检验某概念在当前层是否已线性可分 |
| **3. 稀疏自编码器 (SAE)** | 训练宽度 $M \gg d$ 的过完备稀疏字典 | 无监督特征解耦级 | 解开高维叠加态，提取单义特征方向 |
| **4. 激活修补 (Activation Patching)** | 跨样本移植特定激活并观测输出恢复 | **因果验证黄金标准** | 证明该子空间是决定输出的必要因果机制 |

- **对数几率透镜**（Logit Lens）：无需训练外部参数，直接将第 $\ell$ 层残差流投影至词表空间 $[B, T, V]$，观测各层预测倾向的演进；
- **线性探针**（Linear Probing）：检验特定属性是否在线性子空间中可分，但仅能证明统计相关性，不能证明模型自身在下游使用了该方向；
- **稀疏自编码器**（SAE）：通过 $\hat{\mathbf{x}} = \sum_{k=1}^M f_k(\mathbf{x})\mathbf{d}_k$（$M \gg d$）逆向解开高维叠加态，还原出高纯度的单义特征方向 $\mathbf{d}_k$；
- **激活修补**（Activation Patching）：在“干净输入”与“受损输入”之间定点移植特定层或特定子空间的激活，通过反事实干预（Counterfactual Intervention）确立内部电路的因果作用。

---

至此，我们已经看清了硬盘里 272 个静态权重张量的排布，并亲手追踪了一句文本从 `[B, T]` 整数序列涌入、穿过 30 层 Transformer 直至生成下一个 Token 的全部细节。在下一篇《第 1 篇｜深入 Tokenizer》中，我们将回到阶段 01 的起点，亲手实现把文本切分为 `[B, T]` 整数序列的分词器。
