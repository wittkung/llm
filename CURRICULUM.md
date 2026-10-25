# 《大语言模型与现代 AI 系统：从底层张量到智能体工程》完整课程与系列设计（Rust 白盒实验版）

> **课程设计坐标**：本系列深度融合了 **底层模型白盒解剖（Bottom-Up Tensor Mechanics）**、**哈佛大学 CS2680（*Modern AI Systems: Agents and System Optimizations*，Prof. Juncheng Yang）** 的推理系统与五步实验阶梯（`Mads`），以及 **斯坦福大学 CS146S（*The Modern Software Developer*，Mihail Eric）** 的现代智能体与上下文工程体系。

---

## 一、 为什么要把“底层张量”、“推理系统”和“智能体工程”合在一门课里？

传统大模型课程与教程存在一道巨大的鸿沟：
1. **讲模型原理的教程**往往停留在几年前的玩具级 Python/PyTorch 代码，讲完基础的自注意力（Self-Attention）就戛然而止——读者既不知道现代开源模型（如 `Qwen2.5`、`DeepSeek-V4.1-Flash-1017`）在真实显存里长什么样，也不知道底层张量设计与上层应用有什么关系；
2. **讲 AI 应用与 Agent 开发的教程**往往把大模型当成一个只会收发字符串的 HTTP 黑盒（“Vibe Coding”），一旦遇到长上下文延迟暴增、Token 账单失控、工具调用 JSON 格式崩溃、或者技术面试被追问底层机制，就束手无策。

2026 年最前沿的两门标杆大学课程打破了这道墙：
- **哈佛大学 CS2680（`cs2680.com`）** 敏锐地指出：今天大模型的真实工作负载已经不再是单轮闲聊，而是**长程智能体循环（Agentic Workflows，如 Claude Code、Codex）**。在一次智能体任务中，**95% 以上的计算与显存开销都发生在反复吞吐历史上下文（Prefill 与 KV Cache 前缀复用）**。因此，CS2680 将课程劈为紧密咬合的上下两篇——**Part I 讲智能体架构，Part II 讲支撑智能体的底层推理系统（KV Cache、分页注意力、投机解码、MoE 路由）**，并设计了著名的 **`Mads` 五步实验阶梯**（`Mads-Lens` 观测 $\to$ `Mads-Loop` 循环 $\to$ `Mads-Opt` Token 优化 $\to$ `Mads-Serve` 推理引擎 $\to$ `Mads-Stack` 全栈协同）。
- **斯坦福大学 CS146S（`themodernsoftware.dev`）** 则从现代软件工程视角出发，用 10 周时间系统拆解了 **Coding Agent 的内部循环（Week 1）**、**高级上下文工程与 MCP 协议（Week 2）**、**可复用的 `SKILL.md` 与终端智能体（Week 3）**、**仓库级上下文规约 `CLAUDE.md`/`AGENTS.md` 与子智能体分工（Week 4–5）**，以及 **AI 代码审查、安全防御与异步智能体工厂（Week 6–10）**。

本系列在此基础上更进一步：**向下扎穿最后一层黑盒，用纯 Rust 从磁盘上的 `.safetensors` 字节流开始，一竿子捅到底层的矩阵乘法、中间的 KV Cache 推理引擎、直到顶层的 Coding Agent 与上下文工程。**

---

## 二、 核心教学设计原则（Four Pedagogical Pillars）

1. **先装显影探针，再拆内部零件（The `Lens-First` Principle，借鉴 Harvard CS2680 `Mads-Lens`）**：
   任何一章都不从枯燥的公式推导开始，而是先用 Rust 探针打开真实的工业级文件或运行轨迹，让读者亲眼看到真实的张量维度、字节体积与耗时账单，再解释背后的数学与工程原因。
2. **终端优先、零黑盒依赖、笔记本即可跑通（Terminal-First & Zero-Bloat，借鉴 Stanford CS146S）**：
   彻底告别动辄几十 GB 的 Conda/CUDA 环境配置地狱。全系列基于轻量级纯 Rust 工作区，配合 269 MB 的标准架构模型 `SmolLM2-135M-Instruct`，读者只需 `curl` + `cargo run` 即可在普通笔记本 CPU 上毫秒级复现全部实验。
3. **教学玩具与最前沿旗舰双轨对照（Toy-to-Frontier Dual Track）**：
   每一篇都采用“三阶跳板”：**极简手算玩具（确保能用纸笔算清） $\to$ `SmolLM2-135M` 真实白盒运行（确保能在本地跑通） $\to$ 最前沿旗舰模型（如 `DeepSeek-V4.1-Flash-1017` 的 552B MoE / CED / MLA / 890B KV Cache）架构对照（确保看懂工业界最顶峰的设计）**。
4. **底层机制直接指导上层工程（Systems-Informed Context Engineering）**：
   学完底层每一项机制，立刻回答上层工程问题：学完 Tokenizer（第 1 篇），立刻明白为什么代码缩进和中文提示词要这样写；学完 KV Cache 与 Prefix Tree（第 4 篇），立刻明白为什么设计 Agent System Prompt（第 6 篇）时绝不能把动态时间戳放在静态指令前面。

---

## 三、 全系列三部曲（8 篇长文 + 8 个配套 Rust 实验）完整大纲

全系列分为 **三大模块（模型内核篇、推理系统篇、智能体工程篇）**，共 **8 篇深度长文**，配套同一个持续生长的 Rust 实验工作区：

### 第一部曲（Part I）：模型内核白盒解剖 —— 从字节流到张量流
> **模块目标**：彻底撕掉深度学习框架黑盒，建立从原始字符串、`.safetensors` 静态权重，到每一层动态激活与输出 Logits 的精确张量坐标系。

| 篇目 | 标题与核心命题 | 关键知识点与前沿对标 | 配套 Rust 实验（`crates/`） |
|---|---|---|---|
| **前置篇**<br>（先导篇） | **拆解大模型前的两件趁手兵器：张量线性代数与 Rust 极简入门**<br>*零负担跨越数学与语言门槛* | - **大模型推理仅需的 5 条线性代数法则**：0～4 维张量直觉、向量点积（相似度）、矩阵乘法中间维度抵消律与 `.t()` 转置约定、`matmul` vs. 逐元素乘法 `*`、`reshape / transpose(1,2)` 内存步长与 `.contiguous()`<br>- **Rust 与 `candle` 30 分钟速成**：`let / mut`、`&` 只读借用与 `&mut` 可变借用、`struct / enum / match`、`Option`、`Result` 与 `?` 运算符、PyTorch vs. `candle-core` 算子速查表 | **`Lab 0: llm-tracer` (`--mode prelude`)**<br>- 无需下载模型权重，一键运行 5 组张量线性代数沙盒实验，直观打印形状、点积、门控与内存连续性变化 |
| **第零篇**<br>（总览篇） | **一次大语言模型推理的完整旅程：从输入文本到下一个 Token**<br>*建立全局张量坐标系* | - 硬盘里的模型解剖：`1 + 30×9 + 1 = 272` 个张量（精确到 `134,515,008` 参数与 `269.03 MB`）<br>- **参数矩阵（不含 $B,T$）vs. 中间激活（必含 $B,T$）** 铁律<br>- 11 阶段完整前向计算与 $B=1,T=3,d=4$ 注意力手算<br>- Prefill vs. Decode 形状剧变与 KV Cache 本质<br>- **前沿对标**：`DeepSeek-V4.1-Flash-1017` 的四刀手术（MoE+Engram、20+20 CED、MLA+CSA2+FP4 890B/tok、DSpark）<br>- **认知边界**：无特权基、多义叠加态与 QK/OV 电路分离 | **`Lab 0: llm-tracer`**（对应 `Mads-Lens`）<br>- `--mode inspect`：开箱清点 272 个权重张量<br>- `--mode toy`：核验手算注意力矩阵<br>- `--mode real`：追踪 `SmolLM2-135M` 11 阶段 Prefill + Decode 全过程 |
| **第 1 篇**<br>（分词篇） | **深入 Tokenizer 与对话协议：文本如何变成模型眼中的 `[B, T]` 整数序列**<br>*拆解阶段 01 的零号站台* | - UTF-8 字节流与为什么不能直接按字符/单词切分<br>- Byte-level BPE 训练与推理合并算法（优先队列与合并秩）<br>- 预分词正则演进（GPT-2 $\to$ Llama-3 $\to$ Qwen2.5/DeepSeek 数字与代码切分规则）<br>- Chat Template（ChatML）、角色控制符（`<\|im_start\|>`）与工具调用特殊 Token 注入<br>- 词表大小 $V$ 的工程权衡：为什么小模型用 $49\text{k}$、大模型用 $129\text{k}\sim 262\text{k}$？Token 边界如何制造拼写与算术盲区？ | **`Lab 1: bpe-lens`**<br>- 用 150 行纯 Rust 从零实现字节级 BPE 分词器<br>- 加载 `SmolLM2`、`Qwen2.5` 与 `DeepSeek-V4.1` 的 `tokenizer.json`，可视化对比同一段中文、英文与 Rust 代码的切分粒度与压缩率（Bytes/Token） |
| **第 2 篇**<br>（架构篇） | **深入 Transformer 内部：RoPE、GQA、Pre-RMSNorm、SwiGLU 与 MoE 路由**<br>*拆解阶段 02–08 的算子进化史* | - **位置编码**：为什么绝对位置矩阵被淘汰？复数平面旋转（RoPE）如何让点积只依赖相对距离 $m-n$？GPT-NeoX 半区配对与 `Llama-3.2` 分段波长缩放<br>- **注意力演进**：MHA $\to$ MQA $\to$ GQA $\to$ **DeepSeek MLA**（低秩潜在向量压缩与为什么必须单独剥离 `qk_rope_head_dim=64`）<br>- **归一化与残差主干**：Post-LN 梯度阻塞 vs. Pre-RMSNorm 无阻拦残差直通车；`Gemma-3` 的 `(1+w)` 偏移与 QK-Norm<br>- **前馈网络进化**：双矩阵 FFN $\to$ 三矩阵 SwiGLU $\to$ **细粒度 MoE（共享专家 + 路由专家）与 Engram 稀疏查表记忆** | **`Lab 2: block-anatomy`**<br>- 在 Rust 中独立实现并对比 MHA、GQA 与简化版 MLA 的参数量与 KV 缓存体积<br>- 实测 RoPE 在不同距离 $m-n$ 与不同旋转基数 $\theta$（$10\text{k} \to 100\text{k} \to 500\text{k}$）下的注意力衰减曲线<br>- 用小型路由矩阵模拟 MoE Top-$K$ 门控分发 |

---

### 第二部曲（Part II）：推理系统与算力优化 —— 让每个周期与每个字节算数
> **模块目标**（对标 **Harvard CS2680 Part II: Systems for LLMs**）：从单次前向计算走向高性能推理引擎，彻底弄懂算术强度、显存带宽墙、KV Cache 前缀复用、量化与投机解码。

| 篇目 | 标题与核心命题 | 关键知识点与前沿对标 | 配套 Rust 实验（`crates/`） |
|---|---|---|---|
| **第 3 篇**<br>（采样与解码篇） | **从 `[B, V]` Logits 到受控生成：采样策略、50 微秒语法掩码与投机解码**<br>*拆解阶段 09–11 的出口引擎* | - **概率塑造流水线**：Temperature 如何改变分布熵、Top-$k$ 与 Top-$p$（Nucleus）截断、Min-$p$、Repetition Penalty<br>- **结构化输出（Structured Outputs）**：为什么 Agent 工具调用绝不能靠 Prompt 祈祷 JSON 不出错？剖析微软 `llguidance` / `outlines-core` 如何在每个 Token 间歇（`< 50µs`）将正则表达式/JSON Schema 编译为字节级状态机掩码（Token Mask），将非法 Token 的 Logits 置为 $-\infty$<br>- **打破一步一词串行墙**：投机解码（Speculative Decoding）的无损拒绝采样数学证明，以及 `DeepSeek-V4.1-Flash` 的 DSpark 半自回归草稿机制 | **`Lab 3: constrained-decoder`**<br>- 在 `llm-tracer` 出口实现 Temperature + Top-$p$ 采样器<br>- 用 Rust 实现一个轻量级 JSON 状态机 Logits 掩码器，强制 `SmolLM2-135M` 100% 生成合法的函数调用 JSON<br>- 实现 Draft + Target 投机解码步进验证器，实测一次前向计算验收多个 Token 的加速比 |
| **第 4 篇**<br>（显存与吞吐篇） | **跨越显存墙：Roofline 模型、PagedAttention、前缀缓存复用与量化**<br>*对标 Harvard CS2680 `Mads-Serve`* | - **硬件 Roofline 模型**：定量计算算术强度（FLOPs/Byte），证明为什么 Prefill 是算力受限（Compute-Bound）而 Decode 是显存带宽受限（Memory-Bound）<br>- **KV Cache 显存管理**：为什么连续分配 KV 缓存会浪费 60% 显存？vLLM **PagedAttention** 如何借鉴操作系统虚拟内存分页？<br>- **Prefix Caching（前缀复用）**：SGLang **RadixAttention（基数树缓存）** 如何在多轮对话与 Agent 循环中自动复用历史 KV Cache？<br>- **极致压缩**：从 BF16 $\to$ INT8 $\to$ FP4（E2M1 格式，每 16 通道缩放因子）与 `DeepSeek-V4.1-Flash` 的 CSA2 跨层稀疏索引共享 | **`Lab 4: radix-kv-serve`**（对应 `Mads-Serve`）<br>- 为 `llm-tracer` 编写基于基数树（Radix Tree）的 **Prefix KV Cache 管理器**<br>- 实现 KV Cache 的对称 INT8 / 模拟 FP4 分组量化与反量化，对比精度损失与内存体积变化<br>- 实测多轮对话下开启 Prefix Cache 后的首字延迟（TTFT）下降幅度 |

---

### 第三部曲（Part III）：现代智能体与上下文工程 —— 从推理引擎到自动驾驶软件工程
> **模块目标**（对标 **Stanford CS146S** 与 **Harvard CS2680 Part I & `Mads-Loop / Opt / Stack`**）：用前四篇打下的底层系统视角，自底向上构建并优化现代 Coding Agent（如 Codex CLI、Claude Code）。

| 篇目 | 标题与核心命题 | 关键知识点与前沿对标 | 配套 Rust 实验（`crates/`） |
|---|---|---|---|
| **第 5 篇**<br>（智能体内核篇） | **解剖 Coding Agent：从 `codex-rs` 与 Claude Code 看智能体循环、工具人机工程学与沙箱**<br>*对标 Stanford CS146S Week 1–3 & Harvard `Mads-Loop`* | - **Agent 本质**：揭开智能体神秘面纱——它就是一个包裹在阶段 11 自回归循环外层的 `while (has_tool_call)` 状态机<br>- **工具人机工程学（Tool Ergonomics）**：为什么 `read_file`（带行号切片）、`edit_file`（精确上下文替换而非全文件重写）、`grep`（基于 `ripgrep`）和 `bash` 是 Coding Agent 的四大黄金原语？<br>- **MCP（Model Context Protocol）与 `SKILL.md`**：如何用标准化 JSON-RPC 协议挂载外部工具与按需加载技能说明？<br>- **安全沙箱**：剖析 OpenAI `codex-rs` 如何用 Rust 结合 macOS `seatbelt` / Linux `Landlock` 拦截危险命令 | **`Lab 5: mini-codex-rs`**（对应 `Mads-Loop` & `Mads-Lens`）<br>- 用纯 Rust 从零编写一个 ~300 行的终端 Coding Agent 内核（内置文件读写、差分修改与受限命令执行）<br>- 内置实时轨迹与张量成本监控面板（`Lens`），记录每一轮循环的 Prefill / Cached / Decode Token 数 |
| **第 6 篇**<br>（上下文工程篇） | **让每个 Token 算数：面向 KV Cache 命中的上下文工程、子智能体隔离与仓库规约**<br>*对标 Harvard `Mads-Opt` & Stanford CS146S Week 2, 4, 5* | - **第一性原理级上下文工程（Systems-Aware Context Engineering）**：为什么在 System Prompt 顶部放一个每秒变化的 `current_time` 会瞬间击穿整棵 Radix KV Cache 树（0% 缓存命中）？如何严格按「静态系统规约 $\to$ 仓库 `AGENTS.md` $\to$ 稳定工具定义 $\to$ 追加式对话历史 $\to$ 动态尾部状态」组织上下文层级？<br>- **长程记忆退化与压缩**：当 `T` 增长时注意力分数如何被稀释？滑动窗口裁剪、结构化工作账本与无损摘要何时触发才不破坏 KV Cache？<br>- **子智能体（Subagent）隔离法则**：为什么用只读探索子智能体（Research Subagent）去跑 `grep` 和读文件、只把 200 字结论传回主智能体，能将主干 KV Cache 消耗降低一个数量级？ | **`Lab 6: mads-context-opt`**（对应 `Mads-Opt`）<br>- 搭建 Token 成本与 KV 缓存命中率评测基准（Benchmark）<br>- 对比三种上下文策略：（A）朴素全量拼接 + 顶部动态变量、（B）KV 前缀对齐 + 尾部增量追加、（C）主从 Subagent 上下文隔离，实测完成同一代码重构任务的精确 Token 消耗与缓存命中率 |
| **第 7 篇**<br>（评测、安全与全栈篇） | **构建生产级 AI 系统：从 Terminal-Bench 评测、提示注入防御到全栈智能路由**<br>*对标 Stanford CS146S Week 6–10 & Harvard `Mads-Stack`* | - **现代智能体基准测试**：如何科学衡量模型工程能力？拆解 SWE-bench、Terminal-Bench 4.0、DeepSWE 的评测沙箱设计<br>- **智能体安全护栏（Guardrails & Security）**：间接提示词注入（恶意代码注释或网页诱导 Agent 执行 `curl` 外传密钥）、确定性 Lint/Test Hook 门禁（不通过静态检查禁止提交）<br>- **全栈协同路由（Full-Stack Routing）**：如何模仿 `sgl-router` 与现代 AI 网关，将高频代码补全与简单文件检索路由给本地/轻量 Flash 模型，将复杂架构推理路由给旗舰模型，并实现跨请求 KV 缓存亲和性调度 | **`Lab 7: mads-fullstack`**（对应 `Mads-Stack`）<br>- 组装全系列组件：模型白盒探针 + Radix KV 缓存网关 + 双模型路由器 + 带安全 Hook 的 Rust Agent 循环，跑通一套迷你自动化编程评测任务集 |

---

## 四、 读者学习路径与能力通关地图

通过这三部曲（8 篇长文 + 8 个 Rust 实验），无论是为了**一线 AI 工程开发、系统与算法研究，还是高标准技术面试**，读者都将完成三次认知跃迁：

1. **第一重跃迁（读完 Part I，第 0–2 篇）—— 告别“玄学黑盒”**：
   大脑里装有一张精确到每个维度、每个字节的计算图。看到任何开源模型的 `config.json` 与 `.safetensors`，能立刻算清它的参数分布、GQA/MLA 注意力形状与 RoPE 旋转方式。
2. **第二重跃迁（读完 Part II，第 3–4 篇）—— 告别“算力盲区”**：
   能从硬件内存带宽与算术强度（Roofline）解释为什么 Prefill 和 Decode 性格截然不同，亲手写过结构化 JSON 状态机掩码与基数树 Prefix KV Cache，彻底看懂 vLLM、SGLang 与 `DeepSeek-V4.1-Flash-1017` 的系统优化逻辑。
3. **第三重跃迁（读完 Part III，第 5–7 篇）—— 告别“Vibe Coding”**：
   把底层 KV Cache 物理规律与上层 Agent 工程融会贯通——既能像斯坦福 CS146S 要求的那样设计出高命中率、带安全沙箱与子智能体隔离的现代 Coding Agent，又能像哈佛 CS2680 要求的那样对全栈 Token 成本与推理延迟做定量优化。
