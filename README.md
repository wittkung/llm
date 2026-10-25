# LLM 底层机制与 Rust 白盒追踪系列（Study & Blog Engineering）

本目录用于组织「大语言模型底层机制与 Rust 白盒实验」系列长文的**前期调研（Deep Research）**、**分篇大纲与手算推演**、以及**配套的 Rust 真实模型张量追踪代码**。

---

## 1. 系列核心定位与三部曲课程架构（详见 [`CURRICULUM.md`](CURRICULUM.md)）

本系列深度融合了**底层张量白盒解剖**、**哈佛大学 CS2680（*Modern AI Systems: Agents and System Optimizations*）** 的推理系统与 `Mads` 五步实验阶梯，以及 **斯坦福大学 CS146S（*The Modern Software Developer*）** 的智能体与上下文工程体系，分为 **三大模块（共 8 篇长文 + 8 个配套 Rust 实验）**：

### 第一部曲（Part I）：模型内核白盒解剖 —— 从字节流到张量流
| 篇目 | 标题 | 核心主线与深度边界 | 配套 Rust 实验 |
|---|---|---|---|
| **第零篇（总览篇）** | **一次大语言模型推理的完整旅程：从输入文本到下一个 Token** | 沿着一次真实的前向计算（Forward Pass），清点 `1 + 30×9 + 1 = 272` 个静态权重张量，追踪 11 阶段动态激活形状、玩具手算、Prefill vs Decode 与 KV Cache，并对标最新 `DeepSeek-V4.1-Flash-1017` 的四刀架构手术。 | **`Lab 0: llm-tracer`**（对应 `Mads-Lens`）：支持 `--mode inspect`（开箱体检 272 个权重张量）、`--mode toy`（核验手算矩阵）与 `--mode real`（追踪 `SmolLM2-135M` 11 阶段全过程）。 |
| **第 1 篇（分词篇）** | **深入 Tokenizer 与对话协议：文本如何变成模型眼中的 `[B, T]` 整数序列** | UTF-8 字节流、Byte-level BPE 合并算法、预分词正则演进、ChatML 与工具调用特殊控制符注入、词表大小 $V$ 对显存与压缩率的权衡、Token 边界盲区。 | **`Lab 1: bpe-lens`**：用 150 行 Rust 从零实现 Byte-level BPE，对比 `SmolLM2`、`Qwen2.5` 与 `DeepSeek-V4.1` 的切分粒度与压缩率。 |
| **第 2 篇（架构篇）** | **深入 Transformer 内部：RoPE、GQA、Pre-RMSNorm、SwiGLU 与 MoE 路由** | RoPE 复数旋转推导与分段缩放、MHA $\to$ GQA $\to$ DeepSeek MLA 低秩潜在压缩、Pre-RMSNorm 残差直通车、SwiGLU 三矩阵门控与细粒度 MoE + Engram 查表记忆。 | **`Lab 2: block-anatomy`**：在 Rust 中对比 MHA/GQA/MLA 的 KV 显存占用、RoPE 距离衰减曲线与 MoE Top-$K$ 门控路由。 |

### 第二部曲（Part II）：推理系统与算力优化 —— 让每个周期与每个字节算数（对标 Harvard CS2680 Part II）
| 篇目 | 标题 | 核心主线与深度边界 | 配套 Rust 实验 |
|---|---|---|---|
| **第 3 篇（采样与解码篇）** | **从 `[B, V]` Logits 到受控生成：采样策略、50 微秒语法掩码与投机解码** | Temperature/Top-$k$/Top-$p$ 概率塑造、结构化输出状态机掩码（剖析 `llguidance` 如何在 `<50µs` 屏蔽非法 JSON Token）、投机解码（Speculative Decoding 与 DSpark）打破一步一词串行墙。 | **`Lab 3: constrained-decoder`**：实现完整采样器 + JSON 状态机 Logits 掩码器 + Draft/Target 投机解码批量验证器。 |
| **第 4 篇（显存与吞吐篇）** | **跨越显存墙：Roofline 模型、PagedAttention、前缀缓存复用与量化** | 硬件算术强度（FLOPs/Byte）定量证明 Prefill 计算受限与 Decode 带宽受限、PagedAttention 分页管理、RadixAttention 基数树前缀缓存复用、KV Cache INT8/FP4 量化与 CSA2 稀疏索引。 | **`Lab 4: radix-kv-serve`**（对应 `Mads-Serve`）：实现 Radix Tree 前缀 KV 缓存池与 INT8/FP4 分组量化器，实测多轮对话 TTFT 加速比。 |

### 第三部曲（Part III）：现代智能体与上下文工程 —— 从推理引擎到自动驾驶软件工程（对标 Stanford CS146S & Harvard CS2680 Part I）
| 篇目 | 标题 | 核心主线与深度边界 | 配套 Rust 实验 |
|---|---|---|---|
| **第 5 篇（智能体内核篇）** | **解剖 Coding Agent：从 `codex-rs` 与 Claude Code 看智能体循环、工具人机工程学与沙箱** | 智能体外层状态机循环、`read/edit/grep/bash` 四大工具人机工程学（Tool Ergonomics）、MCP 协议与 `SKILL.md` 按需加载、操作系统级安全沙箱（Seatbelt / Landlock）。 | **`Lab 5: mini-codex-rs`**（对应 `Mads-Loop`）：用纯 Rust 编写 ~300 行终端 Coding Agent 内核与实时 Token 账单监控台。 |
| **第 6 篇（上下文工程篇）** | **让每个 Token 算数：面向 KV Cache 命中的上下文工程、子智能体隔离与仓库规约** | 第一性原理级上下文工程（为什么顶部动态时间戳会击穿整棵 KV Cache 前缀树？）、`AGENTS.md`/`CLAUDE.md` 静态前缀分层、滑动摘要、主从 Subagent 上下文隔离。 | **`Lab 6: mads-context-opt`**（对应 `Mads-Opt`）：对比朴素拼接 vs. KV 前缀对齐 vs. Subagent 隔离的缓存命中率与 Token 成本。 |
| **第 7 篇（评测、安全与全栈篇）** | **构建生产级 AI 系统：从 Terminal-Bench 评测、提示注入防御到全栈智能路由** | SWE-bench / Terminal-Bench 4.0 评测沙箱剖析、间接提示词注入防御与确定性 Lint/Test Hook 门禁、跨模型智能路由与 KV 缓存亲和性调度。 | **`Lab 7: mads-fullstack`**（对应 `Mads-Stack`）：组装探针、Radix KV 缓存、双模型路由与安全 Hook Agent，跑通迷你自动化编程评测集。 |

---

## 1.5 课程设计参考：Stanford CS146S 与 Harvard CS2680 的启发

本系列的实验与篇章演进吸收了两门最新标杆课程的核心设计思想：

1. **Harvard CS2680（*Modern AI Systems: Agents and System Optimizations*，Prof. Juncheng Yang，`cs2680.com`）**：
   - **课程架构**：将“Part I: LLMs & Agents（Claude Code、Agent Workflows、Context Management）”与“Part II: Systems for LLMs（GPU 内存层次、Continuous Batching、KV-Cache & Prefix Reuse、Quantization、Speculative Decoding、MoE Routing）”打通。
   - **五步实验阶梯（`Mads` Lab Series）**：`Mads-Lens`（构建白盒观测探针）$\to$ `Mads-Loop`（构建 Agent 执行循环）$\to$ `Mads-Opt`（Token 级优化与上下文成本压缩）$\to$ `Mads-Serve`（接管开源模型推理引擎）$\to$ `Mads-Stack`（全栈调度优化）。
   - **对本系列的启发**：我们的第零篇配套工具 `llm-tracer` 正对应 **`Lens`（先观测、先给黑盒装上显影探针）**——先用 `--mode inspect` 开箱清点 272 个权重张量，再用 `--mode toy` 与 `--mode real` 跑通 Prefill/Decode，最后直接看懂 `DeepSeek-V4.1-Flash-1017` 为何在 Agent 时代把 Prefill 压到 8B、把 KV Cache 压到 890 Bytes/Token。
2. **Stanford CS146S（*The Modern Software Developer*，Mihail Eric，`themodernsoftware.dev`）**：
   - **课程核心**：告别盲目依赖提示词的“Vibe Coding”，转向 **Agentic Engineering**、**Context Engineering（上下文工程）**、**Spec-Driven Development（规格驱动开发）** 与 **命令行智能体（Terminal Agents）**。
   - **对本系列的启发**：强调**终端优先、零黑盒依赖、可复现的最小实验闭环**——读者不需要配置臃肿的 Python/CUDA 环境，只用 `curl` + `cargo run` 三条终端命令即可在本地笔记本上跑通从 `.safetensors` 字节解析到完整前向推理的全链路实验。

---

## 2. 第零篇的核心验收标准（North Star）

读者读完《第零篇》并跑通配套的 Rust 实验后，应当达到以下明确标准：

> **给我任意一个关键张量，我能说出它的各个维度代表什么、从哪里来、经过哪个参数矩阵、下一步去哪里；并能指出哪些是训练固化的权重（Parameter Matrix），哪些是随本次输入动态产生的激活（Intermediate Activation）。**
>
> **方法论克制**：它能完整解释前向计算的数学与工程机制，但绝不把激活张量中的某一个浮点数直接附会成“模型此刻在想什么”——**维度与数据流完全可追踪，而语义机制仍需要额外的可解释性证据。**

---

## 3. 目录结构规划

```text
llm/
├── Cargo.toml                             # Rust Workspace 顶层配置
├── README.md                              # 全系列蓝图、分工边界与工作流（本文档）
├── CURRICULUM.md                          # 三部曲（8 篇长文 + 8 个 Rust 实验）课程大纲
├── research/
│   ├── deep-research-plan.md              # 4 个前期 Deep Research 课题设计、调研目标与可直接执行的 Prompt
│   └── findings/                          # 存放各 Deep Research 产出的调研报告与数据表
├── posts/
│   ├── ep00-inference-journey/
│   │   ├── outline-and-spec.md            # 第零篇详细写作规格书、玩具手算数值设计、11阶段维度表
│   │   └── draft.md                       # 第零篇正文文稿
│   ├── ep01-tokenizer/                    # 第 1 篇：深入 Tokenizer（后续展开）
│   └── ep02-transformer/                  # 第 2 篇：深入 Transformer（后续展开）
└── crates/
    └── llm-tracer/
        ├── Cargo.toml                     # 第零篇白盒探针 Crate 配置
        ├── DESIGN.md                      # Rust 白盒张量追踪器架构设计与探针输出规范
        └── src/main.rs                    # 探针源码（支持 inspect / toy / real 三模式）
```

---

## 4. 推进工作流建议

1. **阶段一：前期调研（Deep Research）**
   - 阅读 [`research/deep-research-plan.md`](research/deep-research-plan.md)，执行其中设计的 4 个专项调研（特别是**靶机模型微观结构对比**与 **Rust `candle`/`safetensors` 矩阵转置与连续内存布局细节**），将结果沉淀到 `research/findings/`。
2. **阶段二：锁定玩具手算与 Rust 追踪器原型**
   - 基于 [`posts/ep00-inference-journey/outline-and-spec.md`](posts/ep00-inference-journey/outline-and-spec.md) 中的手算矩阵与 [`crates/llm-tracer/DESIGN.md`](crates/llm-tracer/DESIGN.md)，编写 `crates/llm-tracer` 代码，跑出真实的张量形状流水账。
3. **阶段三：撰写与审计第零篇正文**
   - 结合手算推演、维度图（Mermaid / 示意图）与 Rust 真实运行日志，起草《第零篇》长文，并按严肃写作规范完成审计。
