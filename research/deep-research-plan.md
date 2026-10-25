# 前期调研计划（Deep Research Plan）

在动笔撰写《第零篇｜一次大语言模型推理的完整旅程》及开发配套 Rust 追踪器之前，我们需要评估：**哪些内容已经是确定的教科书常识（无需额外调研），哪些内容存在工程细节陷阱、模型间差异或容易误导读者的模糊地带（需要专门的 Deep Research）？**

经过梳理，标准的多头注意力公式本身无需重复调研；但**要把“教学级公式”无缝对接到“用 Rust 追踪一个真实开源模型的 `safetensors` 权重与前向激活”**，有 **4 个核心问题**极易踩坑，非常适合作为前期 Deep Research 课题。

---

## 调研课题总览与必要性评估

| 编号 | 课题名称 | 优先级 | 为什么必须做？（解决什么痛点） | 直接服务于哪部分产出 |
|---|---|---|---|---|
| **DR-01** | **主流轻量级开源 Decoder-Only 模型微观结构与张量命名普查** | **P0（必做）** | 不同开源模型在 `Bias`（偏置项）、`tie_word_embeddings`（词表权重共享）、`head_dim` 比例、GQA 分组比上存在细微但致命的差异。选错靶机或漏看偏置项会导致文中公式与 Rust 打印出的权重列表对不上。 | 第零篇「具体模型形状追踪」选型、Rust 探针加载逻辑 |
| **DR-02** | **Rust 推理生态（`candle` / `safetensors`）内存布局与张量算子精确语义** | **P0（必做）** | `safetensors` 中线性层权重形状是 `[out_dim, in_dim]` 还是 `[in_dim, out_dim]`？`candle` 的 `matmul` 如何处理三维激活 `[B, T, d]` 与二维权重相乘？`transpose` 后为何必须 `.contiguous()`？RoPE 的实部虚部是交错排列（Interleaved）还是对半切分（Half-split）？ | `crates/llm-tracer` 的 200 行白盒实现与博客代码解读 |
| **DR-03** | **经典教科书 Transformer（2017/GPT-2）与现代工业界架构（LLaMA-3/Qwen2.5）的 6 处结构偏移** | **P1（重要）** | 读者脑海里的“位置编码加在 Embedding 后”、“标准 MHA”、“两层 MLP”与真实模型截然不同。需要建立一张精确的“教学模型 vs 真实模型映射表”，防止读者在看懂玩具例子后被真实模型代码绕晕。 | 第零篇「从教学模型到真实模型」过渡节，以及与第 2 篇的边界切分 |
| **DR-04** | **中间激活（Activations）的解释力边界：机械可解释性（Mechanistic Interpretability）前沿共识** | **P1（重要）** | 第零篇文末强调“维度可追踪，但不要把激活中的某个数直接解释成模型在想什么”。需要援引扎实的学术证据（多义神经元 Polysemanticity、叠加态 Superposition、残差流视角、Attention $\neq$ 因果解释），让这一方法论警示有据可查。 | 第零篇结尾「验收标准与认知边界」章节 |

---

## 详细调研规格与 Deep Research Prompts

你可以直接将以下各个模块的 Prompt 复制给 Deep Research 工具（或交由 Agent 拆解执行），并将产出保存至 `research/findings/` 目录下。

---

### DR-01｜主流轻量级开源 Decoder-Only 模型微观结构与张量命名普查
- **建议产出文件**：`research/findings/dr01-small-models-anatomy.md`
- **调研目标**：在 `SmolLM2-135M-Instruct`、`Qwen2.5-0.5B-Instruct`、`Llama-3.2-1B-Instruct`、`Gemma-3-1B-IT` 四个候选模型中，选出最适合作为《第零篇》Rust 追踪靶机的 1 个主模型 + 1 个对照模型。

#### 可直接运行的 Deep Research Prompt：
```markdown
请对以下四个轻量级开源 Decoder-Only 语言模型进行微观架构与权重张量级对比调研：
1. HuggingFaceTB/SmolLM2-135M-Instruct
2. Qwen/Qwen2.5-0.5B-Instruct
3. meta-llama/Llama-3.2-1B-Instruct
4. google/gemma-3-1b-it（或 gemma-2-2b-it）

请直接核实它们在 Hugging Face 上的官方 `config.json`、`model.safetensors.index.json`（或单文件 safetensors header）以及 `transformers` 源码中的 `modeling_*.py`，输出以下精确信息：

1. **核心超参数对比表**：
   - `vocab_size` (V)
   - `hidden_size` (d)
   - `num_hidden_layers` (L)
   - `num_attention_heads` (h_q)
   - `num_key_value_heads` (h_kv) 及 GQA 复用倍率 (g = h_q / h_kv)
   - `head_dim` (d_h，核实是否严格等于 d / h_q，还是像 Gemma 那样独立指定)
   - `intermediate_size` (d_ff)
   - `rms_norm_eps` 与 `rope_theta`（及是否启用了特殊的 rope_scaling）
   - `tie_word_embeddings`（`lm_head.weight` 是否与 `model.embed_tokens.weight` 共享同一块内存，safetensors 文件中是否实际存在 `lm_head.weight` 键）

2. **单层（Layer 0）与全局权重的精确张量清单**：
   - 列出每个模型在 `model.safetensors` 中的真实张量名称（Key）、数据类型（BF16/FP16/FP32）和精确 Shape（按 safetensors 存储顺序 `[dim0, dim1]`）。
   - **重点核查 Bias（偏置项）**：哪些投影层带有 `.bias`（例如 Qwen2.5 的 q_proj/k_proj/v_proj 是否有 bias？SmolLM2 和 Llama-3.2 是否全模型零 bias？）。

3. **特殊架构怪癖（Quirks）排查**：
   - 是否有 Embedding 缩放（如 Gemma 将 Embedding 乘以 `sqrt(hidden_size)`）？
   - 是否有 QK-Norm、Logit Soft-capping 或 Attention 额外缩放因子？
   - Chat Template 注入后，一句最简单的用户提问 `"Hello"` 会被包装成哪几个具体的 Token ID？

4. **教学选型建议**：
   - 如果我们要在博客《第零篇》中向读者展示一个“结构最干净、无多余怪癖、最代表现代 LLaMA 范式、且能在普通笔记本 CPU 上用 Rust 1秒内跑完并打印完整张量形状”的模型，应当选哪一个作为主讲模型？哪一个作为补充对比？
```

---

### DR-02｜Rust 推理生态（`candle` / `safetensors`）内存布局与张量算子精确语义
- **建议产出文件**：`research/findings/dr02-rust-candle-tensor-semantics.md`
- **调研目标**：摸清用 Rust（`candle-core` + `safetensors` + `tokenizers`）手写 200 行透明版前向传播时的所有张量维度转换细节，确保文章里的公式与 Rust 代码 100% 严丝合缝。

#### 可直接运行的 Deep Research Prompt：
```markdown
我们正在用 Rust（基于 Hugging Face 的 `candle-core`、`candle-nn`、`safetensors` 和 `tokenizers` crate）编写一个教学级的白盒 LLM 前向计算追踪器（不调用黑盒模型封装，而是显式编写每一步矩阵运算并在每步之间插入形状与数值探针）。

请结合 `candle` 最新上游源码与文档，调研并回答以下 5 个关键工程与数学对齐问题：

1. **Safetensors 权重存储方向与矩阵乘法语义**：
   - PyTorch 的 `nn.Linear(in_features, out_features)` 在导出到 `safetensors` 时，权重矩阵 `W` 的形状是 `[out_features, in_features]`。
   - 当输入激活 `X` 的形状为 `[B, T, in_features]` 时，在数学正文里我们通常写作 $X W^\top$（其中 $W^\top$ 为 `[in_features, out_features]`）。
   - 在 `candle-core` 中，三维张量 `[B, T, d_in]` 与二维转置张量 `[d_in, d_out]` 直接调用 `.matmul()` 是否会报错（提示维度不匹配）？`candle_nn::Linear::forward` 内部是如何通过 `.broadcast_matmul()` 或 `reshape` 处理 `[B, T, d_in]` 乘 `[d_in, d_out]` 的？

2. **多头拆分（Reshape & Transpose）与内存连续性（Contiguity）**：
   - 将 `[B, T, h * d_h]` 拆分为 `[B, T, h, d_h]` 再 `.transpose(1, 2)` 得到 `[B, h, T, d_h]` 后，张量在内存中变为非连续（non-contiguous stride）。
   - 在 `candle` 中，后续哪些操作（如 RoPE、GQA repeat_kv、`matmul`、合并多头后的 `reshape`）必须显式调用 `.contiguous()`，否则会触发运行时 `Error::RequiresContiguous`？

3. **RoPE（旋转位置编码）的两种内存配对方式**：
   - 业界实现 RoPE 时，维度 $d_h$ 上的两两配对有两种流派：
     (a) GPT-NeoX / HuggingFace Llama 风格：前半截 `[0 .. d_h/2]` 与后半截 `[d_h/2 .. d_h]` 配对（`rotate_half`）；
     (b) 原始 LLaMA (Meta) 风格：相邻奇偶下标 `(2i, 2i+1)` 交错配对。
   - 在 `SmolLM2` / `Llama-3.2` / `Qwen2.5` 的 HuggingFace `safetensors` 权重中，预训练权重对应的是哪一种配对方式？在 `candle_nn::rotary_emb::rope` vs `rope_i` 中应当调用哪一个才能与 HuggingFace Python 输出逐位对齐（误差 < 1e-5）？

4. **GQA（Grouped-Query Attention）的 KV 扩展实现**：
   - 当 $Q$ 为 `[B, h_q, T, d_h]` 而 $K, V$ 为 `[B, h_kv, T, d_h]` 时，如何用最清晰的 3~5 行 `candle` 张量操作（如 `unsqueeze` + `expand` + `reshape`）将 $K, V$ 扩展为 `[B, h_q, T, d_h]`，以便直接执行 $Q K^\top$？

5. **数据类型（BF16 vs F32）与 CPU/Metal 兼容性**：
   - 现代开源模型权重通常以 `BF16` 存储。在 macOS（Apple Silicon CPU 或 Metal 后端）上运行 `candle` 探针时，为了避免 CPU 不支持原生 `BF16` 矩阵乘法并方便打印 `f32` 切片数值，最佳实践是在加载 `safetensors` 时统一 `.to_dtype(DType::F32)` 吗？
```

---

### DR-03｜经典教科书 Transformer 与现代工业界架构的 6 处结构偏移
- **建议产出文件**：`research/findings/dr03-textbook-vs-modern-transformer.md`
- **调研目标**：为《第零篇》和《第 2 篇》整理一份清晰的架构演进对照表，帮读者架起从“标准公式”到“真实模型”的桥梁。

#### 可直接运行的 Deep Research Prompt：
```markdown
在撰写面向开发者的 LLM 原理系列博客时，很多读者熟悉 2017 年《Attention Is All You Need》或 GPT-2 的经典结构，但在阅读现代开源模型（Llama-3、Qwen2.5、Mistral、DeepSeek）的推理代码时，会因为架构演进而产生困惑。

请围绕“一次前向计算的数据流与张量形状”，系统梳理从【经典教科书 Decoder-Only (GPT-2)】到【现代主流开源架构 (Llama-3 / Qwen2.5)】的 6 处核心变化，重点说明：**每一处变化改了哪个公式、有无比重/形状变化、以及为什么在《第零篇（总览篇）》只需知其形状而在《第 2 篇（架构篇）》再深究原理**：

1. **归一化位置与计算方式**：从 Post-LayerNorm（减均值除方差，带 weight 和 bias）到 Pre-RMSNorm（仅均方根缩放，仅带 weight `[d]`，无 bias，且残差主干无阻拦）。
2. **位置信息的注入时机与张量形态**：从在 Embedding 后直接相加绝对位置编码矩阵 `W_pos: [max_seq_len, d]`，变为每层 Attention 内部对拆好头的 `Q, K: [B, h, T, d_h]` 施加旋转位置编码（RoPE）。
3. **注意力头结构**：从 MHA（$h_q = h_{kv}$）到 MQA（$h_{kv}=1$）再到 GQA（$h_q = g \cdot h_{kv}$），权重矩阵 `W_k, W_v` 的形状如何从 `[d, d]` 缩小为 `[h_kv * d_h, d]`？
4. **前馈网络（FFN / MLP）**：从 2 个矩阵（`up: d -> 4d`, `down: 4d -> d` + ReLU/GELU）变为 SwiGLU 的 3 个矩阵（`gate_proj: d -> d_ff`, `up_proj: d -> d_ff`, `down_proj: d_ff -> d`），中间激活张量 `[B, T, d_ff]` 是如何通过逐元素门控相乘汇合的？
5. **偏置项（Bias）的移除趋势**：为什么现代模型普遍去掉了 Linear 层的 Bias？
6. **词表嵌入与输出投影权重共享（Weight Tying）**：什么时候 `embed_tokens.weight` 与 `lm_head.weight` 是同一个矩阵 `[V, d]`，什么时候是分开的两个矩阵？
```

---

### DR-04｜中间激活（Activations）的解释力边界：机械可解释性前沿共识
- **建议产出文件**：`research/findings/dr04-activation-interpretability-limits.md`
- **调研目标**：支撑《第零篇》的收尾方法论——为什么我们能精确追踪每一个张量的维度与数值来源，却不能指着第 5 层隐藏状态里的 `-0.412` 说“这个数代表模型识别出了猫”？

#### 可直接运行的 Deep Research Prompt：
```markdown
我们的博客《第零篇｜一次大语言模型推理的完整旅程》在展示完所有层的权重矩阵与中间激活张量后，最后有一节重要的方法论总结：
“给我任意一个关键张量，我能说出它的各个维度代表什么、从哪里来、经过哪个参数矩阵、下一步去哪里；并能指出哪些是权重，哪些是本次输入产生的激活。它能解释计算机制，但不要把激活中的某个数直接解释成‘模型在想什么’。维度可追踪，语义机制仍需要额外证据。”

请检索机械可解释性（Mechanistic Interpretability）领域的经典文献与共识，为这一节提供严谨、通俗且不夸张的学术支撑（控制在博客读者 3 分钟能读懂的深度）：

1. **残差流（Residual Stream）作为共享通信带宽**：
   - 参考 Elhage et al. (2021) *A Mathematical Framework for Transformer Circuits*，解释为什么每一层输入输出都保持 `[B, T, d]`？残差流如何像一条“主干总线”，各层的 Attention Heads 和 MLP 只是从总线上读取信息并把增量写回（相加）到总线上？
2. **为什么单个激活数值（Single Neuron / Coordinate）不等于单一人类概念？**
   - 简述 **多义性（Polysemanticity）** 与 **叠加假设（Superposition Hypothesis）**（参考 Anthropic 2022 *Toy Models of Superposition*）：当模型要表示的特征数量远大于隐藏维度 $d$ 时，特征并不是正交地对齐在标准坐标轴（单个浮点数下标）上，而是以近似正交的方向组合分布在激活空间中。
3. **为什么注意力权重矩阵（Attention Probs `[B, h, T, T]`）不能直接等同于“模型认为哪个词最重要”？**
   - 简述 *Attention is not Explanation* (Jain & Wallace, 2019) / *Attention is not not Explanation* (Wiegreffe & Pinter, 2019) 以及 Value 向量与 Output 投影 (`W_V`, `W_O`) 的作用：即使某个位置的注意力权重 $\alpha_{ij}$ 很大，如果该位置对应的 $\|V_j W_O\|$ 范数极小或方向被正交抵消，它对残差流的实际贡献也可能微乎其微。
4. **如果需要真正的语义证据，研究者通常怎么做？（点到为止，留作拓展视野）**
   - 简述线性探针（Linear Probing）、激活修补/因果干预（Activation Patching / Causal Scrubbing）、稀疏自编码器（Sparse Autoencoders, SAE）以及 Logit Lens（直接把中间层 `[B, T, d]` 乘上 `lm_head` 观察词表分布演化）各自在做什么。
```
