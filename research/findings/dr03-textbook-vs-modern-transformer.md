# **从经典 GPT-2 到现代 Llama-3 与 Qwen2.5：基于单次前向计算数据流与张量形状的架构演进全景解析**

## **架构演进的张量契约与教学解耦逻辑**

在面向开发者的生成式大语言模型（LLM）原理系列技术专栏中，许多具备深度学习基础的读者对 2017 年《Attention Is All You Need》或 GPT-2 的经典 Decoder-Only 架构耳熟能详。然而，当开发者打开 Hugging Face transformers 库或 vLLM、SGLang 等现代推理引擎中关于 Llama-3、Qwen2.5、Mistral 或 DeepSeek 的源码（如 modeling\_qwen2.py 或 llama3.py）时，往往会在类定义与算子调用细节上产生强烈的割裂感。这种困惑并非源于 Transformer 的核心范式发生了颠覆，而是因为工业界在过去数年间围绕**训练稳定性（Training Stability）**、显存带宽瓶颈（Memory Bandwidth Bottleneck）**与**长上下文外推（Long-Context Extrapolation）三大系统工程目标，对模型内部的微观算子进行了彻底的重写。  
要消除这种认知断层，最有效的教学架构是将“宏观张量数据流”与“微观算子数学原理”进行分层解耦。在整座 Transformer 大厦中，无论内部组件如何演进，**单次前向计算（Forward Pass）的主干张量形状契约（Tensor Shape Contract）始终保持严格不变**：输入形状为 \[B, T\]（Batch Size 为 B，序列长度为 T）的离散词元索引 input\_ids，经词表嵌入层映射为 \[B, T, d\]（隐藏层维度为 d）的连续隐状态张量 hidden\_states；随后该张量依次穿过 L 个结构同构的 Decoder 层，在每一层的入口与出口处始终维持 \[B, T, d\] 的形状不变；最终经由输出投影层 lm\_head 转化为 \[B, T, V\]（词表大小为 V）的对数几率张量 logits。  
下表系统对比了从经典教科书架构（GPT-2）到现代主流开源架构（Llama-3 / Qwen2.5）在单次前向计算中的六处核心演进、算子公式更迭、张量形状变化以及在系列博客中的教学分层策略：

| 核心演进维度 | 经典教科书架构 (GPT-2) | 现代主流开源架构 (Llama-3 / Qwen2.5) | 权重与中间激活张量形状变化 | 《第零篇（总览篇）》关注点（黑盒形状契约） | 《第 2 篇（架构篇）》深究点（白盒数学与系统原理） |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **1\. 归一化位置与计算** | Post-LN（或早期 Pre-LN）：减均值除标准差，含 weight 与 bias | **Pre-RMSNorm**：残差分支内前置均方根缩放，仅含 weight，无 bias | 移除偏置参数 \[d\]，仅保留缩放参数 \[d\]；输入输出激活维持 \[B, T, d\] | 视为形状守恒的逐位置缩放算子：\[B, T, d\] \-\> \[B, T, d\]，认识残差主干的直通拓扑 | 推导梯度消失/爆炸的残差高速路机制、去均值冗余性证明及 GPU 访存算子融合收益 |
| **2\. 位置信息注入** | 绝对位置编码：Embedding 后一次性相加可学习矩阵 W\_pos | **旋转位置编码 (RoPE)**：每层 Attention 内部对拆头后的 Q, K 施加复数旋转 | 移除参数矩阵 \[max\_seq\_len, d\]；在每层对 \[B, h, T, d\_h\] 注入 \[B, 1, T, d\_h\] 的正余弦张量 | 明确位置信息不再于入口相加，而是随 position\_ids 传入每层 Attention 内部作用于 Q 和 K | 推导二维旋转矩阵如何通过内积仅依赖相对距离 m-n，以及 rope\_theta 调频与长文本外推原理 |
| **3\. 注意力头结构** | 多头注意力 (MHA)：查询头数等于键值头数 (h\_q \= h\_{kv} \= h) | 分组查询注意力 (GQA)：h\_q \= g \\cdot h\_{kv}，g 个查询头共享一组键值头 | W\_k, W\_v 权重从 \[d, \[span\_19\](start\_span)\[span\_19\](end\_span)\[span\_27\](start\_span)\[span\_27\](end\_span)d\] 缩小为 \[h\_kv \* d\_h, d\]；相乘前 K, V 经广播对齐至 \[B, h\_q, T, d\_h\] | 掌握 Q 与 \[span\_35\](start\_span)\[span\_35\](end\_span)K, V 投影通道数的非对称拆分，以及计算注意力分数前 K, V 的头维度对齐规则 | 结合 Roofline 模型剖析自回归解码的内存墙，量化 KV Cache 显存压缩比与吞吐量提升 |
| **4\. 前馈网络 (FFN)** | 双矩阵 MLP：up (d \-\> 4d) \+ GELU \+ down (4d \-\> d) | **三矩阵 SwiGLU**：gate (d \-\> d\_ff) 与 up (d \-\> d\_ff) 逐元素门控相乘后接 down | 2 个矩阵变为 3 个矩阵；双分支激活 \[B, T, d\_ff\] 经哈达玛积汇合为单一 \[B, T, d\_ff\]；d\_{ff} 调整为约 \\frac{8}{3}d 变体 | 追踪激活张量从单一路径分叉为两条 \[B, T, d\_ff\] 支路、逐元素相乘汇合再降维回 \[B, T, d\] 的流向 | 剖析平滑门控（SiLU）对非线性表达力的增强，推导三矩阵等参数量下的 \\frac{8}{3}d 缩放与硬件对齐法则 |
| **5\. 线性层偏置项** | 全部保留：所有 Linear 层与 LayerNorm 均带可学习 bias | **普遍移除 (bias=False)**：Llama-3 全移除；Qwen2.5 仅在 QKV 保留 bias=True | 线性层移除形状为 \[d\_out\] 的一维偏置参数向量；前向激活张量形状无任何改变 | 了解线性投影退化为纯矩阵乘法 X W^T（或带广播加法），完全不影响三维激活张量维度 | 解释 Pre-RMSNorm 与 RoPE 下偏置的数学冗余、深层网络数值漂移抑制，以及 Qwen 系列 QKV 偏置与 QK-Norm 的演进 |
| **6\. 词表与输出共享** | 权重绑定 (Weight Tying)：输入嵌入与输出投影共享同一矩阵 | **按模型规模分化**：小模型（0.5B/1.5B）绑定；中大模型（\\ge 7\\text{B}）解绑为独立矩阵 | 绑定时共享单一 \[V, d\] 矩阵；解绑时 embed\_tokens 与 lm\_head 各自独立持有 \[V, d\] 矩阵 | 认识首尾两端 \[B, T\] \-\> \[B, T, d\] 与 \[B, T, d\] \-\> \[B, T, V\] 的对称形状转换契约 | 计算超大词表（V \\approx 15.2\\text{万}）在不同模型尺寸下的参数占比账本，剖析输入/输出语义空间解耦与流水线并行收益 |

从单层 Decoder 的计算图拓扑结构来审视，经典教科书架构与现代开源架构在内部数据流布局上呈现出显著的结构性重组。经典设计将 LayerNorm 直接置于残差主干之上，在模型入口处一次性叠加绝对位置编码，采用通道完全对称的多头注意力（MHA）带偏置线性投影，并串联一个由升维与降维两块矩阵构成的 GELU 前馈网络。相比之下，现代 Llama-3 与 Qwen2.5 架构将 Pre-RMSNorm 移入侧支入口，从而在整座网络中开辟出一条毫无阻拦的残差直通主干（Residual Trunk）；在注意力支路内部，键值投影被分组查询注意力（GQA）显著收窄，而旋转位置编码（RoPE）则直接注入到每一层拆分好头的查询与键张量之中；在紧随其后的前馈网络支路中，特征流先分叉进入并行的门控投影（Gate Projection，后接 SiLU 激活）与升维投影（Up Projection），经逐元素乘法汇合后再由降维投影（Down Projection）回归残差主干。  
这种“《第零篇（总览篇）》重张量形状、《第 2 篇（架构篇）》重数学与系统原理”的编排设计，本质上是遵循软件工程中的**接口与实现分离原则（Separation of Interface and Implementation）**。对于初次接触现代大模型源码的开发者而言，如果在一开始就陷入复数域旋转矩阵推导、矩阵乘法算子融合或分布式显存通信的细节，极易迷失在局部公式中而丧失对全局前向计算图的掌控。只要在《第零篇》中锚定每一处张量形状的“契约接口”，读者就能在脑海中建立起一条畅通无阻的 \[B, T, d\] 数据流水线；待全局骨架确立后，再于《第 2 篇》逐一拆解每个算子模块内部的代数恒等变换与硬件优化动因，便能水到渠成地实现从“看懂代码形状”到“洞悉架构灵魂”的跃迁。

## **变化一：归一化位置与计算方式（从 Post-LayerNorm 到 Pre-RMSNorm）**

### **算子公式演进：从后置均值方差归一化到前置均方根缩放**

在 2017 年的原始 Transformer 架构中，归一化层采用的是**后置层归一化（Post-LayerNorm）**，即先将子层（自注意力或前馈网络）的输出与残差分支相加，再在残差主干上执行归一化。以单层子模块函数 F(\\cdot) 表示 Attention 或 FFN，Post-LayerNorm 的计算公式为：  
x\_{l+1} \= \\text{LayerNorm}(x\_l \+ F(x\_l))  
其中，经典 LayerNorm 对隐藏层维度 d 上的向量 x \\in \\mathbb{R}^d 计算均值 \\mu 与方差 \\sigma^2，通过中心化与缩放后施加可学习的仿射变换参数 \\gamma \\in \\mathbb{R}^d（代码中的 weight）和 \\beta \\in \\mathbb{R}^d（代码中的 bias）：  
\\mu \= \\frac{1}{d}\\sum\_{i=1}^d x\_i, \\quad \\sigma^2 \= \\frac{1}{d}\\sum\_{i=1}^d (x\_i \- \\mu)^2, \\quad \\text{LayerNorm}(x) \= \\frac{x \- \\mu}{\\sqrt{\\sigma^2 \+ \\epsilon}} \\odot \\gamma \+ \\beta  
从 GPT-2 开始，为了改善深层网络的训练收敛性，架构演进迈出了第一步：将归一化位置从残差主干移至子层分支的入口处，形成 **Pre-LayerNorm** 结构。而到了现代主流开源架构（Llama-3、Qwen2.5、Mistral、DeepSeek-V3），不仅彻底固化了前置（Pre-Norm）拓扑，还将归一化计算方式全面替换为 **RMSNorm（Root Mean Square Layer Normalization）**。在 Qwen2DecoderLayer 等现代模块中，每层包含 input\_layernorm 和 post\_attention\_layernorm 两个实例，其完整单层前向递推公式演进为：  
h\_l \= x\_l \+ \\text{Attention}(\\text{RMSNorm}\_1(x\_l)), \\quad x\_{l+1} \= h\_l \+ \\text{FFN}(\\text{RMSNorm}\_2(h\_l))  
其中，RMSNorm 彻底摒弃了均值 \\mu 的计算与减法中心化操作，也不再使用偏置项 \\beta，仅计算输入向量各分量的均方根（RMS）并进行缩放：  
\\text{RMS}(x) \= \\sqrt{\\frac{1}{d}\\sum\_{i=1}^d x\_i^2 \+ \\epsilon}, \\quad \\text{RMSNorm}(x) \= \\frac{x}{\\text{RMS}(x)} \\odot \\gamma  
公式中的 \\epsilon 为防止除零的下溢保护常数（在 config.json 中体现为 rms\_norm\_eps，例如 Qwen2.5-0.5B 设为 10^{-6}，而 Qwen2.5-14B 与 32B 为增强大规模训练下的数值鲁棒性将其调大至 10^{-5}）。此外，在 Gemma-3 与最新演进的 Qwen3 等前沿架构中，RMSNorm 还进一步被引入到注意力模块内部的查询与键向量上（即 **QK-Norm**），在施加位置编码前对每个注意力头的 Q 和 K 进行独立归一化，以抑制超大规模预训练后期的注意力对数几率爆炸（Attention Logit Growth）。

### **参数比重与张量形状变化**

在张量形态与参数量维度上，从 LayerNorm 到 RMSNorm 的演进极其干净，输入与输出激活张量形状严格守恒。无论是 LayerNorm 还是 RMSNorm，其归约操作（Reduction）仅发生在最后一个特征维度 d（即 dim=-1）上。输入张量 x 的形状为 \[B, T, d\]，内部计算出的统计量 rms 形状为 \[B, T, 1\]，经广播除法与逐元素乘法后，输出张量形状依然严格为 \[B, T, d\]。  
与此同时，可学习参数削减了一半，且占总模型比重极小。经典 LayerNorm 拥有 weight: \[d\] 和 bias: \[d\] 两个一维向量（共 2d 个参数）；而现代 RMSNorm 移除了 bias，仅保留单一的 weight: \[d\]（共 d 个参数）。以 Qwen2.5-7B（L=28, d=3584）为例，全模型共有 2L \+ 1 \= 57 个 RMSNorm 层（每层 2 个，外加进入 lm\_head 前的 1 个最终归一化层），总参数量仅为 57 \\times 3584 \= 204,288（约 20.4 万参数），占 70 亿级总参数的比重不足 0.003\\%。

### **为什么在《第零篇》只需知其形状而在《第 2 篇》再深究原理**

在《第零篇（总览篇）》中，读者的核心任务是打通整个前向传播的张量拼图。由于 RMSNorm 是一个完全不改变张量维度（\[B, T, d\] \-\> \[B, T, d\]）的逐元素缩放算子，且参数占比可以忽略不计，若在总览篇展开均值方差消除或反向传播链式法则，会严重打断读者对主干数据流的追踪。因此，在《第零篇》只需让读者建立两个直观认知：一是它是 \[B, T, d\] 形状守恒的数值稳定器；二是它位于残差支路的入口处，使得主干道 x \= x \+ sublayer(norm(x)) 成为一条没有任何算子阻拦的直通恒等映射路径。  
而在《第 2 篇（架构篇）》中，则必须从**优化动力学**与**硬件系统工程**两个深层维度剖析这一演进的必然性。首先是**残差主干的梯度高速路（Gradient Highway）效应**。在 Post-LN 结构中，每一层的主干输出都要经过一次非线性归一化缩放，反向传播时梯度必须连乘 L 次 LayerNorm 的雅可比矩阵，导致深层网络的底层梯度呈指数级衰减，极易在训练初期崩溃，高度依赖严苛的学习率预热（Warm-up）。而在 Pre-RMSNorm 结构下，将全部 L 层展开可得 x\_L \= x\_0 \+ \\sum\_{l=0}^{L-1} F\_l(\\text{RMSNorm}(x\_l))。反向传播时，损失函数对任意浅层状态 x\_0 的偏导数恒包含一项单位矩阵 \\frac{\\partial x\_L}{\\partial x\_0} \= I \+ \\dots，这意味着梯度可以跨越数十乃至上百层畅通无阻地回传至词嵌入层，极大提升了百亿与千亿参数模型的训练稳定性。  
其次是**算子级访存带宽（Memory Bandwidth）优化**。根据 Zhang & Sennrich (2019) 的实证研究，LayerNorm 的重中心化（减去均值 \\mu）对模型的平移不变性贡献甚微，真正起关键作用的是均方根带来的尺度不变性（Scaling Invariance）。在 GPU 硬件执行层面，归一化属于典型的访存受限型（Memory-Bound）算子而非计算受限型算子。传统 LayerNorm 需要在寄存器或共享内存中对向量 x 进行两次归约扫描（或维护复杂的 Welford 在线方差状态）以同时求取均值和方差；而 RMSNorm 仅需一次平方和归约即可完成计算，不仅减少了约 40\\% 的算术指令，更大幅降低了 GPU 高带宽显存（HBM）与片上高速缓存（SRAM）之间的往返读写开销，使算子耗时显著下降。

## **变化二：位置信息的注入时机与张量形态（从绝对位置矩阵相加到注意力层内 RoPE）**

### **算子公式演进：从入口处绝对位置向量加法到层内复数域相对角度旋转**

自注意力机制本身是排列不变的（Permutation-Invariant），必须显式注入位置信息才能感知词元顺序。在 GPT-2 等经典架构中，位置信息通过可学习的绝对位置编码（Learned Absolute Positional Embeddings）在模型最底部的词嵌入阶段一次性注入。设词元嵌入矩阵为 W\_{\\text{embed}} \\in \\mathbb{R}^{V \\times d}，绝对位置编码矩阵为 W\_{\\text{pos}} \\in \\mathbb{R}^{T\_{\\max} \\times d}，对于位置索引为 m \\in \\{0, 1, \\dots, T-1\\} 的输入词元 x\_m，其初始隐状态 h\_m^{(0)} 的计算公式为直接逐元素相加：  
h\_m^{(0)} \= W\_{\\text{embed}}\[x\_m\] \+ W\_{\\text{pos}}\[m\]  
此后，位置信息便混合在隐状态向量中，随残差流穿过全部 L 层，后续各层不再显式接收位置索引。  
现代主流开源架构（Llama-3、Qwen2.5、Mistral、DeepSeek）彻底废弃了底部的 W\_pos 加法，转而在**每一层 Attention 模块内部**，对刚刚通过线性投影并拆分好注意力头的查询向量 q\_m \\in \\mathbb{R}^{d\_h} 和键向量 k\_n \\in \\mathbb{R}^{d\_h} 施加**旋转位置编码（Rotary Position Embedding, RoPE）**，而值向量 v\_n 则完全不参与位置旋转。  
RoPE 的核心公式是将头维度 d\_h（必为偶数，如 64 或 128）两两分组为 \\frac{d\_h}{2} 个二维子空间（对应复平面上的向量），对第 i \\in \\{0, 1, \\dots, d\_h/2 \- 1\\} 个二维分量 (x^{(2i)}, x^{(2i+1)})，根据其所处的绝对序列位置 m 乘以二维旋转矩阵，旋转角度为 m\\theta\_i：  
\\text{RoPE}(x\_m, m) \= R\_{\\Theta, m}^{d\_h} x\_m \= x\_m \\odot \\c\[span\_21\](start\_span)\[span\_21\](end\_span)\[span\_29\](start\_span)\[span\_29\](end\_span)os(m\\Theta) \+ \\tilde{x}\_m \\odot \\sin(m\\Theta)  
在 Hugging Face 的 modeling\_qwen2.py 与 modeling\_llama.py 实现中，为避免将偶数位和奇数位交错排布带来的内存重排开销，通常采用前半段与后半段配对（Half-Split Rotation）的等价实现：设向量 x\_m \= \[x\_m^{(1)}, x\_m^{(2)}\]（其中 x\_m^{(1)}, x\_m^{(2)} \\in \\mathbb{R}^{d\_h/2} 分别为前一半和后一半通道），定义辅助翻转函数 \\text{rotate\\\_half}(x\_m) \= \[-x\_m^{(2)}, x\_m^{(1)}\]，则预计算好的正余弦张量与 q, k 的逐元素结合公式为：  
q\_m^{\\text{rot}} \= q\_m \\odot \\cos(m\\Theta) \+ \\text{rotate\\\_half}(q\_m) \\odot \\sin(m\\Theta) k\_n^{\\text{rot}} \= k\_n \\odot \\cos(n\\Theta) \+ \\text{rotate\\\_half}(k\_n) \\odot \\sin(n\\Theta)  
其中，基准频率序列 \\Theta \= \\{\\theta\_i \= \\text{base}^\[span\_36\](start\_span)\[span\_36\](end\_span){-2i/d\_h} \\mid i \\in \[0, d\_h/2)\\}。早期 Llama-1/2 采用经典基准值 \\text{base} \= 10,000；而在现代长上下文模型中，rope\_theta（即 \\text{base}）被大幅拉升：例如 Llama-3 将其设为 500,000，并在 Llama-3.1 中引入包含 factor、low\_freq\_factor 与 high\_freq\_factor 的分段频率缩放（rope\_scaling）；Qwen2.5 与 Qwen3 则通过 ABF（Adjusted Base Frequency）技术将 rope\_theta 统一提升至 1,000,000.0，以支撑 32K 至 128K 的原生上下文窗口。

### **参数比重与张量形状变化**

这一机制的变革在模型权重字典（state\_dict）与前向激活张量流向中引发了深远变化。首先是**可学习位置权重矩阵彻底归零**。经典 GPT-2 中形状为 \[max\_seq\_len, d\] 的参数矩阵 wpe.weight 被完全移除。如果一个现代支持 128K 上下文的模型（如 Qwen2.5-32B，T\_{\\max}=131,072, d=5120）仍采用可学习绝对位置编码，仅 W\_pos 一项就将凭空吞噬 131,072 \\times 5120 \\approx 6.71 \\times 10^8（约 6.71 亿）个可训练参数。而 RoPE 的频率 \\theta\_i 是由超参数决定的确定性三角函数值，不含任何可训练参数（参数量为 0）。  
其次是**激活张量的作用时机与四维广播对齐**。在模型入口处，embed\_tokens(input\_ids) 输出 \[B, T, d\] 后直接进入第 0 层，不再有任何加法操作。在每一层 Attention 内部，q\_proj 与 k\_proj 投影并拆头转置后，得到四维张量 Q: \[B, h\_q, T, d\_h\] 与 K: \[B, h\_kv, T, d\_h\]。此时，rotary\_emb 模块根据 position\_ids: \[B, T\] 生成形状为 \[B, T, d\_h\] 的 cos 和 sin 张量；在 apply\_rotary\_pos\_emb 函数中，通过指定 unsqueeze\_dim=1 将正余弦张量扩维为 \[B, 1, T, d\_h\]，利用广播机制同时作用于所有 h\_q 个查询头和 h\_{kv} 个键头，输出的 Q\_rot 与 K\_rot 形状依然严格保持为 \[B, h\_q, T, d\_h\] 与 \[B, h\_kv, T, d\_h\]。此外，尽管现代模型主干普遍采用 bfloat16 精度运行，但在 Qwen2RotaryEmbedding 等实现中，位置角度 m\\theta\_i 及正余弦表均强制在 float32（全精度）下计算，待旋转完成后再转回 bfloat16，以防止当序列位置 m 达到数万乃至十万量级时，低精度浮点数因尾数位不足而发生严重的位置混叠（Position Aliasing）与信息丢失。

### **为什么在《第零篇》只需知其形状而在《第 2 篇》再深究原理**

在《第零篇（总览篇）》中，读者在阅读模型顶层 forward 代码时，会发现 position\_i\[span\_38\](start\_span)\[span\_38\](end\_span)ds 并没有在 embed\_tokens 后面被消费掉，而是被一路传递进每一个 DecoderLayer 的 self\_attn 中，或者在顶层生成一对 (cos, sin) 元组广播给所有层。此时，读者只需要掌握**数据流的注入坐标与广播形状**：即位置信息是在 Q 和 K 变成 \[B, h, T, d\_h\] 之后、计算 QK^T 之前，通过与 \[B, 1, T, d\_h\] 的正余弦张量做逐元素乘加注入的，且完全不改变 Q 和 K 的张量形状，也不触碰 V。这足以让读者毫无障碍地读懂 KV Cache 更新前后的张量拼接逻辑（因为写入 KV Cache 的必须是已经施加过 RoPE 的 K\_rot 和原始的 V）。  
至于为什么非要用如此繁琐的三角函数旋转来替代简单的矩阵相加，则必须留到《第 2 篇（架构篇）》进行严密的数学推导。第一个核心原理是**通过绝对位置操作实现纯粹的相对位置感知**。在注意力机制中，决定词元 m 对词元 n 关注程度的是内积 \\langle q\_m^{\\text{rot}}, k\_n^{\\text{rot}} \\rangle。由于正交旋转矩阵满足性质 (R\_{\\Theta, m}^{d\_h})^T R\_{\\Theta, n}^{d\_\[span\_39\](start\_span)\[span\_39\](end\_span)h} \= R\_{\\Theta, n-m}^{d\_h}，旋转后的内积在代数上严格展开为仅依赖相对位置差 m-n 的函数：  
\\langle R\_{\\Theta, m}^{d\_h} q\_m, R\_{\\Theta, n}^{d\_h} k\_n \\rangle \= q\_m^T R\_{\\Theta, n-m}^{d\_h} k\_n  
这意味着模型无需像 T5 或 ALiBi 那样显式构造并维护庞大的 \[T, T\] 相对位置偏置矩阵（这会破坏许多算子优化），仅靠对单个位置向量施加线性复杂度的 O(T \\cdot d) 旋转，就在内积空间中天然获得了平移不变的相对位置编码能力。  
第二个核心原理是**长上下文外推与远程衰减特性**。在绝对位置编码下，一旦推理序列长度超过训练时的 max\_seq\_len，超出部分的 W\_pos\[m\] 根本未被训练过，模型输出立即崩溃；此外，将位置编码在底层与词嵌入相加后，经过数十层非线性变换与残差混合，位置信号在深层会严重退化。而 RoPE 在每一层 Attention 中重新注入干净的几何相位，且高频分量负责局部精细语序、低频分量负责长程语义关联，自然呈现出“距离越远、内积上限越低”的远程衰减包络，为后续通过调整 rope\_theta、YaRN（Yet another RoPE extensioN）与双块注意力（Dual Chunk Attention, DCA）实现从 4K 到 128K 乃至百万级上下文的无损外推奠定了坚实的频域数学基础。

## **变化三：注意力头结构演进（从 MHA 到 MQA 再到 GQA）**

### **算子公式演进：从全独立键值头到分组共享键值投影**

在经典《Attention Is All You Need》与 GPT-2 的多头注意力（Multi-Head Attention, MHA）中，查询（Query）、键（Key）和值（Value）拥有完全相同的注意力头数，即 h\_q \= h\_k \= h\_v \= h，且每个头的维度为 d\_h \= d / h。对于第 i \\in \\{1, \\dots, h\\} 个注意力头，其拥有完全独立的投影矩阵 W\_Q^{(i)}, W\_K^{(i)}, W\_V^{(i)} \\in \\mathbb{R}^{d \\times d\_h}，单头注意力计算公式为：  
\\text{head}\_i \= \\text{Softmax}\\left(\\frac{(X W\_Q^{(i)}) (X W\_K^{(i)})^T}{\\sqrt{d\_h}} \+ M\\right) (X W\_V^{(i)})\[span\_5\](start\_span)\[span\_5\](end\_span)\[span\_8\](start\_span)\[span\_8\](end\_span)\[span\_11\](start\_span)\[span\_11\](end\_span)  
其中 M 为因果掩码矩阵（Causal Mask）。所有 h 个头的输出在特征维度拼接后，乘以输出投影矩阵 W\_O \\in \\mathbb{R}^{d \\times d}。  
随着大模型参数量与上下文长度的激增，自回归解码（Autoregressive Decoding）阶段的 KV Cache 显存占用与带宽读取成为头号系统瓶颈。为此，Shazeer (2019) 提出了极端压缩的**多查询注意力（Multi-Query Attention, MQA）**，令所有 h\_q 个查询头共享唯一的一组键值头（即 h\_{kv} \= 1）。然而，对于百亿参数以上的大模型，将 h\_{kv} 直接砍到 1 会导致显著的表达能力退化与训练不稳定。  
现代主流开源架构（Llama-3 全系列、Qwen2.5 全系列、Mistral-7B）普遍收敛到了折中的黄金标准——**分组查询注意力（Grouped-Query Attention, GQA）**。GQA 将 h\_q 个查询头均匀划分为 h\_{kv} 个组（满足整除关系 g \= h\_q / h\_{kv}，其中 g 为每组包含的查询头数，通常满足 1 \< h\_{kv} \< h\_q）。同一组内的 g 个查询头各自拥有独立的查询投影矩阵 W\_Q^{(j, r)} \\in \\mathbb{R}^{d \\times d\_h}（组索引 j \\in \\{1, \\dots, h\_{kv}\\}，组内偏移 r \\in \\{1, \\dots, g\\}），但**共同共享该组唯一的一套键投影矩阵 W\_K^{(j)} \\in \\mathbb{R}^{d \\times d\_h} 和值投影矩阵 W\_V^{(j)} \\in \\mathbb{R}^{d \\times d\_h}**。于是，第 (j, r) 个查询头的计算公式变为：  
\\text{head}\_{j, r} \= \\text{Softmax}\\left(\\frac{\\text{RoPE}(X W\_Q^{(j, r)}) \\cdot \\text{RoPE}(X W\_K^{(j)})^T}{\\sqrt{d\_h}} \+ M\\right) (X W\_V^{(j)})  
可以看到，MHA 是 g \= 1（即 h\_{kv} \= h\_q）时的特例，而 MQA 是 g \= h\_q（即 h\_{kv} \= 1）时的特例。此外，在追求极致超长上下文效率的 DeepSeek-V2/V3 架构中，还演化出了另一条技术路线——**多头潜在注意力（Multi-head Latent Attention, MLA）**，通过将键值联合压缩至低秩潜在向量 c\_{KV} \\in \\mathbb{R}^{d\_c}（其中 d\_c \\ll\[span\_22\](start\_span)\[span\_22\](end\_span)\[span\_30\](start\_span)\[span\_30\](end\_span) h\_{kv} d\_h）并在推理时利用矩阵乘法结合律将解压矩阵吸收到 W\_Q 和 W\_O 中，实现了比 GQA 更高的 KV Cache 压缩率与更强的表达力 。

### **参数比重与张量形状变化**

在 PyTorch 源码（如 Qwen2Attention.\_\_init\_\_）中，nn.Linear(in\_featur\[span\_96\](start\_span)\[span\_96\](end\_span)es, out\_features) 的内部权重张量 .weight 形状定义为 \[out\_features, in\_features\]。从 MHA 到 GQA，线性投影层与中间激活张量的形状发生了极其鲜明的非对称收缩。具体而言，q\_proj.weight 维持 \[h\_q \* d\_h, d\]（即 \[d, d\]）不变，o\_proj.weight 同样维持 \[d, h\_q \* d\_h\]（即 \[d, d\]）不变；而 k\_proj.weight 与 v\_proj.weight 则双双从经典的 \[d, d\] 缩小为 **\[h\_kv \* d\_h, d\]**，缩减倍数恰好为查询分组比 g \= h\_q / h\_{kv}。在 vLLM 等高性能推理引擎中，为了减少 CUDA Kernel 启动开销，会将这三个矩阵在输出通道维度上拼接融合为一个单一的 qkv\_proj.weight，其形状由 MHA 时代的 \[3 \* d, d\] 演进为 **\[(h\_q \+ 2 \* h\_kv) \* d\_h, \[span\_40\](start\_span)\[span\_40\](end\_span)d\]**。  
下表展示了 GQA 架构在 Qwen2.5 各尺寸模型中的具体权重张量形状缩减账本：  
\| 模型名称 \| 隐藏维度 d \| 查询头数 h\_q \| 键值头数 h\_{kv} \| 头维度 d\_h \| 分组比 g \| q\_proj.weight 形状 \| k\_proj.weight / v\_proj.weight 形状 \| \| :--- \| :--- \| :--- \| :--- \| :--- \| :--- \| :--- \| :--- \| \| **Qwen2.5-0.5B** \| 896 \| 14 \| 2 \| 64 \| 7:1 \| \[896, 896\] \| **\[128, 896\]**（缩减为 1/7） \| \| **Qwen2.5-1.5B** \| 1536 \| 12 \| 2 \| 128 \| 6:1 \| \[1536, 1536\] \| **\[256, 1536\]**（缩减为 1/6） \| \| **Qwen2.5-7B** \| 3584 \| 28 \| 4 \| 128 \| 7:1 \| \[3584, 3584\] \| **\[512, 3584\]**（缩减为 1/7） \|  
单看 Attention 模块内部的参数量，在 g=7 的配置下，四个投影矩阵的总参数量从 4d^2 降至 (2 \+ \\frac{2}{7})d^2 \\approx 2.286 d^2，削减了约 43\\% 的注意力层参数；而节省下来的参数预算，在现代架构中通常被转移并投资到了更大容量的 SwiGLU 前馈网络中。  
在前向计算的数据流中，这种非对称结构需要通过特定的张量形状对齐机制（repeat\_kv）来衔接。输入 hidden\_states: \[B, T, d\] 经过线性映射后，得到 q: \[B, T, h\_q \* d\_h\] 以及较窄的 k, v: \[B, T, h\_kv \* d\_h\]。经 view 拆头与 transpose(1, 2\) 后，Q 的形状为 **\[B, h\_q, T, d\_h\]**，而 K 和 V 的形状为 **\[B, h\_kv, T, d\_h\]**（在施加 RoPE 并更新 KV Cache 时，缓存中存储的也是这个较小形状的 K 和 V）。在执行标准的批量矩阵乘法 torch.matmul(Q, K.transpose(2, 3)) 之前，由于第 1 维的头数不匹配（h\_q \\neq h\_{kv}），在 Hugging Face 的原生实现（Eager Attention）中会调用 repeat\_kv(hidden\_states, n\_rep=g) 函数：该函数通过 unsqueeze(2)、.expand(B, h\_kv, g, T, d\_h) 和 .res\[span\_118\](start\_span)\[span\_118\](end\_span)hape(B, h\_q, T, d\_h)，将每个 KV 头在组内复制 g 份，使 K 和 V 的形状在逻辑上扩展为 \[B, h\_q, T, d\_h\]，从而算出 \[B, h\_q, T, T\] 的注意力权重矩阵并加权求和得到 \[B, h\_q, T, d\_h\]，最后经转置合并与 o\_proj 还原为 \[B, T, d\]。而在 FlashAtten\[span\_42\](start\_span)\[span\_42\](end\_span)tion-2 或 SDPA 等底层融合算子中，则无需在显存中真正物理复制 K, V 张量，而是通过 CUDA 线程块索引映射（令第 i 个 Query 头直接按整除索引 \\lfloor i / g \\rfloor 去读取对应的 KV 头），实现零额外显存开销的隐式广播。

### **为什么在《第零篇》只需知其形状而在《第 2 篇》再深究原理**

在《第零篇（总览篇）》中，初学者阅读 modeling\_qwen2.py 或 modeling\_llama.py 时最容易卡壳的地方，就是看到 k\_proj 和 v\_proj 的输出维度与 q\_proj 不一致，随后又出现一个看似陌生的 repeat\_kv 函数。此时，只要从**张量维度匹配契约**的角度说明：“为了节省显存，K 和 V 少算了几倍的头（只有 h\_{kv} 个头）；为了能跟拥有 h\_q 个头的 Q 做矩阵乘法，在相乘前把每个 K, V 头复制 g 份凑齐 h\_q 个头”，读者就能瞬间打通维度变换的逻辑闭环，顺利走完 Attention 的形状推导。  
而在《第 2 篇（架构篇）》中，则必须引入计算机体系结构中的 **Roofline 模型（算术强度与内存墙分析）来揭示 GQA 成为现代大模型标配的深层系统原因。大语言模型的推理由两个截然不同的阶段组成：预填充阶段（Prefill Phase）与自回归解码阶段（Decode Phase）。在 Prefill 阶段，输入序列的 T 个词元并行计算，矩阵乘法涉及大量的数据复用，属于计算受限（Compute-Bound）**；但在 Decode 阶段，每次前向传播仅输入 1 个新生成的词元（此时查询张量形状为 \[B, h\_q, 1, d\_h\]），却需要从显存中读取历史全部 T 个位置的 KV Cache（形状为 \[B, h\_kv, T, d\_h\]）来计算注意力。此时算术强度（每字节显存读取对应的浮点运算次数 FLOPs/Byte）跌至约等于 1\\sim 2，远低于现代 GPU（如 NVIDIA H800/A100）数百 FLOPs/Byte 的平衡点，推理速度彻底被显存带宽（Memory Bandwidth）卡死。  
通过引入 GQA 将键值头数压缩为原来的 1/g（例如 Qwen2.5-7B 压缩为 1/7，Llama-3-8B 从 32 头压缩为 8 头即 1/4），每一层每生成一个词元所需从 HBM 读取的 KV Cache 数据量直接成比例缩减了 g 倍：  
\\text{KV Cache 字节数} \= 2 \\times B \\times T \\times L \\times h\_{kv} \\times d\_h \\times \\text{Bytes}(\\text{dtype})  
这不仅使单卡在 128K 超长上下文场景下能够容纳数倍于 MHA 的并发批次（Batch Size），更直接把 Decode 阶段的显存带宽耗时砍掉了 75\\%\\sim 85\\%，而模型在各类基准评测集上的困惑度（Perplexity）与下游任务精度几乎与全量 MHA 持平。

## **变化四：前馈网络结构与激活门控（从双矩阵 GELU 到三矩阵 SwiGLU）**

### **算子公式演进：从单路双层感知机到双路门控线性单元**

在经典 Transformer 与 GPT-2 中，前馈网络（FFN / MLP）是一个极其简单的**串联两层全连接网络**：先通过升维矩阵 W\_{\\text{up}} \\in \\mathbb{R}^{d \\times 4d} 将特征维度从 d 扩展至 4d，经过逐元素非线性激活函数（2017 原版为 ReLU，GPT-2 为 GELU），再通过降维矩阵 W\_{\\text{down}} \\in \\mathbb{R}^{4d \\times d} 投影回维度 d：  
\\text{FFN}\_{\\text{classic}}(x) \= \\text{GELU}(x W\_{\\text{up}} \+ b\_{\\text{up}}) W\_{\\text{down}} \+ b\_{\\text{down}}  
而在现代主流开源架构（Llama-3、Qwen2.5、Mistral、DeepSeek 的稠密层及 MoE 专家层）中，FFN 全面演进为基于 **SwiGLU（Swish-Gated Linear Unit）** 的**三矩阵门控拓扑**。以 Qwen2MLP 的实现为例，该模块定义了三个无偏置的线性投影矩阵：门控投影 gate\_proj（权重 W\_{\\text{gate}} \\in \\mathbb{R}^{d \\times d\_{ff}}）、升维投影 up\_proj（权重 W\_{\\text{up}} \\in \\mathbb{R}^{d \\times d\_{ff}}）以及降维投影 down\_proj（权重 W\_{\\text{down}} \\in \\mathbb{R}^{d\_{ff} \\times d}）。其前向计算公式为：  
\\text{FFN}\_{\\text{SwiGLU}}(x) \= \\Big(\\text{SiLU}(x W\_{\\text{gate}}) \\odot (x W\_{\\text{up}})\\Big) W\_{\\text{down}}  
其中，算子 \\odot 表示逐元素相乘（Hadamard Product / Element-wise Multiplication）；激活函数 \\text{SiLU}(\\cdot)（Sigmoid Linear Unit，又称 Swish-1，在 config.json 中标记为 "hidden\_act": "silu"）定义为自门控平滑函数：

\\text{SiLU}(z) \= z \\cdot \\sigma(z) \= \\frac{z}{1 \+ e^{-z}}

### **参数比重与张量形状变化**

从经典 MLP 到 SwiGLU MLP，数据流在模块内部经历了一次“Y 字形分叉与汇合”，同时中间维度 d\_{ff}（即配置中的 intermediate\_size）的取值规则也发生了深刻变化。在单次前向计算中，输入张量 x（即经过 post\_attention\_layernorm 后的隐状态）形状为 **\[B, T, d\]**。进入模块后首先发生**分叉**：x 同时送入两个并行的线性层，第一条支路计算 gate \= self.gate\_proj(x)，形状变为 **\[B, T, d\_ff\]**，紧接着原地施加 act\_fn（即 \\text{SiLU}），形状保持 **\[B, T, d\_ff\]**；第二条支路计算线性映射 up \= self.up\_proj(x)（不加任何激活函数），形状同样为 **\[B, T, d\_ff\]**。随后进入**汇合阶段**：两条支路的张量通过逐元素乘法 intermediate \= silu(gate) \* up 汇合。由于两个操作数的形状完全一致（均为 \[B, T, d\_ff\]），逐元素相乘既不改变维度也不涉及跨通道归约，汇合后的中间激活张量形状依然严格为 **\[span\_108\](start\_span)\[span\_108\](end\_span)\[span\_112\](start\_span)\[span\_112\](end\_span)\[B, T, d\_ff\]**。最后在**降维阶段**，汇合后的张量通过 output \= self.down\_proj(intermediate)，从 \[B, T, d\_ff\] 投影还原为 **\[B, T, d\]**，并加回到残差主干上。在 vLLM 的 Qwen2MLP 实现中，由于 gate\_proj 和 up\_proj 的输入完全相同（均为 x），引擎在加载权重时会将 gate\_proj.wei\[span\_109\](start\_span)\[span\_109\](end\_span)\[span\_113\](start\_span)\[span\_113\](end\_span)ght: \[d\_ff, d\] 和 up\_proj.weight: \[d\_ff, d\] 沿第 0 维拼接为一个融合权重矩阵 **gate\_up\_proj.weight: \[2 \* d\_ff, d\]**，前向计算时仅执行一次矩阵乘法得到 \[B, T, 2 \* d\_ff\]，再通过融合 CUDA 算子 SiluAndMul 将前一半通道经 SiLU 激活后与后一半通道直接相乘，一步输出 \[B, T, d\_ff\]。  
在中间层维度 d\_{ff} 的缩放与参数比重演变上，经典双矩阵 FFN 的中间维度为 4d，单层参数量为 2 \\times (d \\times 4d) \= 8d^2。而在三矩阵 SwiGLU 中，单层 FFN 参数量为 3 \\times (d \\times d\_{ff}) \= 3 d \\cdot d\_{ff}。为了在不增加单层总参数量的前提下引入第 3 个矩阵，Noam Shazeer (2020) 提出将 d\_{ff} 从 4d 缩减为原来的 \\frac{2}{3}，即 d\_{ff} \= \\frac{8}{3}d \\approx 2.67\[span\_124\](start\_span)\[span\_124\](end\_span)\[span\_126\](start\_span)\[span\_126\](end\_span)d（此时 3 \\times d \\times \\frac{8}{3}d \= 8d^2，参数量严格持平）。然而，在现代结合了 GQA 的开源模型中，由于 GQA 大幅削减了 Attention 模块的参数量，架构师往往会将省下来的参数预算补贴给 FFN，使 d\_{ff} 介于 2.67d 到 5.8d 之间，并向上取整为 128 或 256 的整数倍以对齐 GPU Tensor Core 硬件分块。例如，在 **Qwen2MoE** 基础配置中，d \= 2048, d\_{ff} \= 5632，膨胀比为 5632 / 2048 \= 2.75\\times（接近经典的 \\frac{8}{3}d）；而在 **Qwen2.5** 稠密系列中，为了强化单层知识存储容量，膨胀比被设定得更高：Qwen2.5-0.5B（d=896 \\to d\_{ff}=4864，约 5.43\\times）、Qwen2.5-1.5B（d=1536 \\to d\_{ff}=8960，约 5.83\\times）、Qwen2.5-7B（d=3584 \\to d\_{ff}=18944，约 5.29\\times） 以及 Qwen2.5-32B（d=5120 \\to d\_{ff}=27648，恰好为 5.4\\times）。在这种现代配置下，单层 SwiGLU FFN 的参数量（例如 Qwen2.5-7B 单层 FFN 参数量为 3 \\times 3584 \\times 18944 \\approx 2.036 \\times 10^8）占到了整个 Decoder Layer 参数量（约 2.22 \\times 10^8）的 **91.7\\%**，意味着现代稠密大模型超过八成到九成的层内权重参数都集中在 SwiGLU 的三个矩阵之中 。

### **为什么在《第零篇》只需知其形状而在《第 2 篇》再深究原理**

在《第零篇（总览篇）》中，读者看到 FFN 从两行代码变成了 down\_proj(act\_fn(gate\_proj(x)) \* up\_proj(x))，最核心的困惑在于“为什么会有两个一模一样的升维矩阵并行计算，然后乘在一起”。在总览阶段，只需引导读者看清**双支路同形汇合的张量拓扑**：两个 \[d, d\[span\_157\](start\_span)\[span\_157\](end\_span)\[span\_160\](start\_span)\[span\_160\](end\_span)\[span\_163\](start\_span)\[span\_163\](end\_span)\_ff\] 的矩阵各自把 \[B, T, d\] 映射成 \[B, T, d\_ff\]，中间那个星号 \* 只是同形状张量的逐位置相乘（不改变 \[B, T, d\_ff\] 的形状），然后再由 down\_proj 压回 \[B, T, d\]。同时提醒读者不要死记硬背教科书里的 4d，而是直接读取配置文件中的 intermediate\_size 即可。  
而在《第 2 篇（架构篇）》中，则需要深入揭示 SwiGLU 的数学机制与硬件工程细节。首先是**双线性门控（Bilinear Gating）与动态特征路由**。在传统单路 MLP \\text{GELU}(x W\_{\\text{up}}) 中，每个神经元只能通过单一线性超平面加固定激活函数来筛选特征；而在 SwiGLU 中，\\text{SiLU}(x W\_{\\text{gate}}) 充当了一个由输入动态决定的软门控（Soft Gate），控制着另一条纯线性信息通路 x W\_{\\text{up}} 中每个特征通道的放行比例。从泰勒展开视角看，两条含参线性分支的逐元素相乘引入了输入特征之间的**二阶乘法交互项（Multiplicative Interactions）**，极大地增强了单层网络拟合复杂逻辑关系与键值记忆（Key-Value Memory）的能力。同时，在反向传播时，由于 up 分支不经过激活函数，\\frac{\\partial (\\text{SiLU}(g) \\odot u)}{\\partial g} 和 \\frac{\\partial (\\text{SiLU}(g) \\odot u)}{\\partial u} 提供了互补的双重梯度通路，有效缓解了深层网络中的神经元坏死问题。  
其次是**硬件张量核心（Tensor Core）维度对齐与算子融合法则**。在《第 2 篇》中可以向读者展示，为什么经典公式推导出的 \[span\_44\](start\_span)\[span\_44\](end\_span)\\frac{8}{3}d 在实际工程中总是变成看似奇怪的数字（如 18944 或 27648）。这是因为现代 GPU 的矩阵乘法单元（Tensor Cores）在矩阵维度是 128 或 256 的整数倍时才能达到峰值吞吐率（避免 Tile 边界填充浪费），因此工业界均采用 \\lceil \\frac{8}{3}d / 256 \\rceil \\times 256 或特定倍数对齐的公式来最终敲定 intermediate\_size。

## **变化五：线性层偏置项（Bias）的移除趋势与局部保留特例**

### **算子公式演进：从仿射变换到纯线性映射及 QKV 特例**

在经典神经网络与 GPT-2 中，所有的全连接层（c\_attn, c\_proj, c\_fc）均执行标准仿射变换（Affine Transformation），即在矩阵乘法后加上可学习的偏置向量 b \\in \\mathbb{R}^{d\_{\\text{out}}}：  
y \= x W^T \+ b  
而在以 PaLM、Llama-1/2/3、Mistral 与 DeepSeek 为代表的现代主流架构中，所有的线性投影层（包括 Attention 中的 q\_proj, k\_proj, v\_proj, o\_proj 以及 FFN 中的 gate\_proj, up\_proj, down\_proj）和最终的 lm\_head，在实例化 nn.Linear 时均显式传入 bias=False（在配置中对应 attention\_bias: false 和 mlp\_bias: false），算子公式彻底简化为纯线性映射：  
y \= x W^T  
然而，在引导开发者阅读现代开源代码时，必须重点指出一个极具代表性的**工业界架构特例——Qwen2 与 Qwen2.5 系列**。如果开发者打开 modeling\_qwen2.py 源码，会发现一种精心设计的混合策略：在 Qwen2MLP 的 gate\_proj、up\_proj、down\_proj 以及 Qwen2Attention 的输出投影 **o\_proj** 中，全部遵循现代惯例设置了 **bias=False**；但在 Qwen2Attention 的查询、键、值三个输入投影层（**q\_proj、k\_proj、v\_proj**）中，却刻意保留了 **bias=True**（在衍生配置中常被称为 qkv\_bias=True）。其局部公式为：  
Q \= X W\_Q^T \+ b\_Q, \\quad K \= X W\_K^T \+ b\_K, \\quad V \= X W\_V^T \+ b\_V, \\quad O \= \\text{Attention}(Q, K, V) W\_O^T  
更进一步地，在随后的架构代际演进中，**Qwen3** 团队在技术报告中明确指出：为了在超大规模训练下获得更极致的稳定性，Qwen3 最终**移除了 Qwen2/2.5 中保留的 QKV-bias**，并同步引入了前文所述的 **QK-Norm** 来接管注意力分布的数值稳定职能。这一“全保留（GPT-2）\\to 全移除（Llama-3/Mistral）与局部保留 QKV 偏置（Qwen2.5）并存 \\to 移除 QKV 偏置并替换为 QK-Norm（Qwen3）”的演进轨迹，是向开发者展示真实工业界架构权衡的绝佳范本。

### **参数比重与张量形状变化**

在线性层中，矩阵乘法 X W^T 的输出形状为 \[B, T, d\_out\]。当存在偏置向量 bias: \[d\_out\] 时，系统通过广播机制将其视为 \[1, 1, d\_out\] 并逐元素加到 \[B, T, d\_out\] 上。因此，无论 bias 是 True 还是 False，**前向传播中任何节点的激活张量形状都完全没有任何改变**。  
在参数量维度上，偏置项所占比重微乎其微。一个形状为 \[d\_out, d\_in\] 的权重矩阵拥有 d\_{\\text{out}} \\times d\_{\\text{in}} 个参数，而对应的偏置向量仅有 d\_{\\text{out}} 个参数，两者相差数千倍（即 1 / d\_{\\text{in}}）。以 Qwen2.5-7B（L=28, d=3584, h\_q=28, h\_{kv}=4, d\_h=128）为例，每层保留的 q\_proj.bias: \[3584\]、k\_proj.bias: \[512\] 与 v\_proj.bias: \[512\] 合计仅有 3584 \+ 512 \+ 512 \= 4608 个参数，全模型 28 层累计仅 129,024（约 12.9 万）个参数，占总模型参数的比重仅为 **0.0017\\%** 。即便把 FFN 和 o\_proj 的偏置也全部加回来，全模型也不过增加约 125 万参数（占比约 0.016\\%）。显然，移除 Bias 绝不是为了“节省显存容量”。

### **为什么在《第零篇》只需知其形状而在《第 2 篇》再深究原理**

在《第零篇（总览篇）》中，既然 Bias 的有无既不改变激活张量形状 \[B, T, d\_out\]，其参数量占比又在万分之一以下，那么在总览篇只需要用一句话向读者交底：“现代模型普遍在 nn.Linear 中设置 bias=False（仅 Qwen2.5 等少数模型在 q\_proj/k\_proj/v\_proj 保留了 bias=True），这只是省去了矩阵乘法后的一个同形状向量广播加法，对整体数据流形状毫无影响”。这样读者在对照 Llama-3 和 Qwen2.5 的权重键名（为什么 Qwen2.5 的 safetensors 文件里多了 self\_attn.q\_proj.bias 而 Llama-3 没有）时就不会感到疑惑。  
而在《第 2 篇（架构篇）》中，则可以从三个深层机制剖析为什么现代模型（尽管偏置参数极少）依然坚定地走向“去 Bias 化”，以及 Qwen2.5 为什么曾在 QKV 上留了一手。第一，**Pre-RMSNorm 下的代数冗余与均值漂移抑制**。在传统神经网络中，偏置项 b 的核心作用是调整激活函数（如 ReLU）的触发阈值，或者补偿输入特征的非零均值偏移。然而在现代架构中，一方面 FFN 的输出直接汇入残差主干 x\_{l+1} \= h\_l \+ \\text{FFN}(h\_l)，并在进入下一层时立即被 Pre-RMSNorm 重新缩放；另一方面，Chowdhery et al. (PaLM, 2022\) 发现，在百亿参数以上、采用 bfloat16 混合精度训练的极深网络中，线性层的独立偏置项 b 容易在多层残差累加中产生与输入无关的常数分量累积（Activation Drift），诱发训练中后期的损失尖峰（Loss Spikes）。去掉线性层 Bias 迫使所有特征变换严格通过零点（Origin-Preserving），显著提升了大规模训练的数值稳定性。  
第二，**张量并行（Tensor Parallelism）与算子调度精简**。在 Megatron-LM 风格的分布式张量并行中，o\_proj 和 down\_proj 采用行并行（Row-Parallel Linear），各 GPU 计算出局部矩阵乘法结果后需执行一次跨卡 All-Reduce 求和。如果带有 bias，要么只能在 All-Reduce 之后单独启动一个加法算子（打断通信与计算流水线），要么将 bias 除以并行度 TP 后在各卡局部相加（容易引入低精度浮点舍入误差）。彻底移除 bias=False 使得所有线性层退化为纯粹的矩阵乘法（GEMM），不仅消除了分布式规约中的边界处理，还避免了在小 Batch 解码阶段为加载一维偏置向量而额外消耗宝贵的寄存器与 L1 Cache 带宽。  
第三，**Qwen2.5 保留 qkv\_bias 的长文本外推考量与 Qwen3 的最终演进**。既然其它地方都去掉了 Bias，为何 Qwen1/Qwen2/Qwen2.5 唯独在 Q, K, V 上保留了偏置？设 q\_m \= x\_m W\_Q^T \+ b\_Q，k\_n \= x\_n W\_K^T \+ b\_K，在施加 RoPE 旋转后计算注意力内积时，展开式中除了词元内容交互项 x\_m W\_Q^T R\_{\\Theta, n-m} W\_K x\_n^T 外，还会产生由偏置项主导的交叉项（如 b\_Q R\_{\\Theta, n-m} (x\_n W\_K^T)^T 与 b\_Q R\_{\\Theta, n-m} b\_K^T）。相关理论分析与实证研究表明，这些与输入词元内容无关、仅依赖相对旋转角度 n-m 的偏置内积项，相当于在注意力分数矩阵上隐式叠加了一层类似 ALiBi 的**可学习相对位置先验偏置**，能够有效增强模型的长上下文外推能力（Length Extrapolation），并在中小规模模型训练中平抑注意力损失尖峰。而到了 Qwen3 时代，随着预训练词元量突破 36 万亿（36T tokens）且全面引入了专门的 32K 长上下文训练阶段、ABF 调频、YaRN 与双块注意力（DCA），模型不再需要依赖 qkv\_bias 来辅助外推；相反，移除 qkv\_bias 并换上 QK-Norm 能够从根源上锁死查询与键向量的模长，防止超大模型在长周期训练中出现注意力熵坍缩（Attention Entropy Collapse）。

## **变化六：词表嵌入与输出投影的权重共享机制（Weight Tying 的分水岭）**

### **算子公式演进：从强制共享单一矩阵到按模型规模动态解耦**

在 Decoder-Only 语言模型的首尾两端，存在着两个对称的维度转换操作。在模型入口（Token Embedding）处，系统根据输入词元索引序列 X \\in \\{0, \\dots, V-1\\}^{B \\times T}，从词表嵌入矩阵 W\_{\\text{embed}} \\in \\mathbb{R}^{V \\times d} 中按行查表（Lookup），得到初始隐状态 H^{(0)} \\in \\mathbb{R}^{B \\times T \\times d}。在模型出口（LM Head Projection）处，经过 L 层 Decoder 处理并完成最终 RMSNorm 后，得到末端隐状态 H^{(L)} \\in \\mathbb{R}^{B \\times T \\times d}，将其与输出投影矩阵 W\_{\\text{lm\\\_head}} \\in \\mathbb{R}^{V \\times d} 的转置相乘，得到词表空间上的未归一化对数几率 \\text{Logits} \\in \\mathbb{R}^{B \\times T \\times V}：  
\\text{Logits} \= H^{(L)} W\_{\\text{lm\\\_head}}^T  
在经典的 GPT-2（以及 T5、BERT 等早期模型）中，为了压缩参数量并遵循 Press & Wolf (2017) 提出的正则化假说，普遍采用了**权重绑定（Weight Tying，在 config.json 中对应 "tie\_word\_embeddings": true）**，即强制令输入嵌入矩阵与输出投影矩阵指向显存中的**同一个物理张量**：  
W\_{\\text{lm\\\_head}} \\equiv W\_{\\text{embed}}  
而在现代主流开源架构（Llama-3、Qwen2.5、Mistral、DeepSeek）中，这一机制不再是“一刀切”，而是演化出了一条清晰的**基于模型参数规模与词表大小的工程分水岭**。具体而言，当配置为 tie\_word\_embeddings: true 时（主要见于 **3B 及以下的轻量级/端侧现代模型**，如 **Qwen2.5-0.5B**、**Qwen2.5-1.5B**），代码在模型初始化末尾执行 self.lm\_head.weight \= self.model.embed\_tokens.weight，两者共享同一个 \[V, d\] 矩阵；而当配置为 tie\_word\_embeddings: false 时（普遍见于 **7B 及以上的主流中大型现代模型**，如 **Qwen2.5-7B / 14B / 32B**、**Llama-3.1-8B / 405B**、**DeepSeek-V3**），model.embed\_tokens.weight 与 lm\_head.weight 则是两个完全独立初始化、独立更新的 \[V, d\] 矩阵。

### **参数比重与张量形状变化**

在 PyTorch 的底层内存布局上，nn.Embedding(V, d) 的 .weight 形状为 **\[V, d\]**；而 nn.Linear(d, V, bias=False)（即 lm\_head）的 .weight 形状因为遵循 \[out\_features, in\_features\] 规范，同样恰好为 **\[V, d\]**。正是因为这两个模块在底层内存布局上的二维形状完全一致（均为 \[V, d\]），才使得它们可以直接共享同一个底层数据指针（Data Pointer）而无需在每次前向传播时发生物理转置复制。当共享时（tie\_word\_embeddings: true），首尾两端合计仅占用 **1 \\times V \\times d** 个参数；当解绑时（tie\_word\_embeddings: false），首尾两端合计占用 **2 \\times V \\times d** 个参数。  
为什么现代模型会在 0.5B/1.5B 与 7B/32B 之间发生如此整齐的 true 到 false 的翻转？核心原因在于**现代多语言与代码分词器（Tokenizer）的词表规模 V 相比 GPT-2 暴增了 3 倍**。GPT-2 的词表大小仅为 V \= 50,257；而现代多语言模型如 Qwen2.5 的词表高达 V \= 151,936（在 7B/14B/32B 中为对齐硬件分块填充为 V \= 152,064）。  
下表基于 Qwen2.5 系列各尺寸模型的真实配置参数，精确测算了单个 \[V, d\] 矩阵的参数体量及其对 tie\_word\_embeddings 决策的直接支配作用：

| 模型名称 | 隐藏层维度 d (hidden\_size) | 词表大小 V (vocab\_size) | 单个 \[V, d\] 矩阵参数量 | 若不绑定（2 \\times V \\times d）占总模型参数比重 | 实际配置 tie\_word\_embeddings | 架构决策背后的参数比重逻辑 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Qwen2.5-0.5B** | 896 | 151,936 | **1.36 亿** (136.1\\text{M}) | 约 **55.1\\%**（2.72 亿 / 4.94 亿） | **true（绑定）** | 若解绑，首尾两个投影层将吞噬全模型过半参数，严重挤压 24 层主干网络的容量 |
| **Qwen2.5-1.5B** | 1536 | 151,936 | **2.33 亿** (233.4\\text{M}) | 约 **30.2\\%**（4.67 亿 / 15.4 亿） | **true（绑定）** | 单个词表矩阵已占总参数 15.1\\%，绑定可省下 2.33 亿参数全部投入 28 层 Transformer 主干 |
| **Qwen2.5-7B** | 3584 | 152,064 | **5.45 亿** (545.0\\text{M}) | 约 **14.3\\%**（10.90 亿 / 76.1 亿） | **false（解绑）** | 参数预算充足，解绑新增的 5.45 亿参数占比降至约 7\\%，换取输入与输出空间的自由解耦 |
| **Qwen2.5-32B** | 5120 | 152,064 | **7.79 亿** (778.6\\text{M}) | 约 **4.8\\%**（15.57 亿 / 325 亿） | **false（解绑）** | 64 层深层网络主干占绝对主导（\>95\\%），独立训练 lm\_head\[span\_135\](start\_span)\[span\_135\](end\_span) 对模型表达力与分布式部署更优 |

### **为什么在《第零篇》只需知其形状而在《第 2 篇》再深究原理**

在《第零篇（总览篇）》中，从前向计算的数据流视角来看，无论 e\[span\_143\](start\_span)\[span\_143\](end\_span)mbed\_tokens.\[span\_144\](start\_span)\[span\_144\](end\_span)weig\[span\_137\](start\_span)\[span\_137\](end\_span)ht 与 lm\_head.weight 在底层显存里是不是同一个张量，它们执行的算子契约完全不变：入口处接收 \[B, T\] 查表输出 \[B, T, d\]，出口处接收 \[B, T, d\] 乘以 \[V, d\]^T 输出 \[B, T\[span\_151\](start\_span)\[span\_151\](end\_span)\[span\_155\](start\_span)\[span\_155\](end\_span), V\]。因此，在总览篇只需向读者交代两者的形状同为 \[\[span\_145\](start\_span)\[span\_145\](end\_span)V, d\] 且互为转置对称操作，并顺带提示：“在小模型里这两个层常常共用同一份权重以节省显存，而在 7B 以上的大模型里则是两个独立的矩阵”。这样读者在检查模型权重文件（为什么 0.5B 的 model.safetensors 里找不到 lm\_head.weight 这个键，或者在 LoRA 合并导出时需要特别注意 tie\_word\_embeddings 配置） 时便能迅速定位原因。  
而在《第 2 篇（架构篇）》中，除了算清楚上表所示的“15.2 万超大词表参数占比账本”之外，还应当从**表征学习理论**与**分布式系统并行**两个维度深究为什么当参数预算充裕（\\ge 7\\text{B}）时，现代模型纷纷放弃了权重绑定。第一，**输入语义编码与输出概率判别的空间解耦**。在模型入口处，embed\_tokens 的任务是将离散的词元符号映射到一个紧凑的初始语义空间，使得近义词或语法功能相似的词元在进入第 0 层前具有相近的几何表示；而在模型出口处，经过了数十层非线性变换与最终的 RMSNorm 之后，末端隐状态 H^{(L)} 所处的特征流形已经与第 0 层的输入流形截然不同，lm\_head 的任务是充当一个高维多分类判别器（Classifier），最大化目标词元与干扰词元之间的分类间隔。强行将这两个处于计算图首尾两端的矩阵绑死在一起，会导致同一个矩阵在反向传播时同时承受来自“第 0 层底层特征提取”和“末端交叉熵分类损失”两股方向迥异的梯度拉扯，限制了大模型在海量词元训练下的收敛上限。  
第二，**流水线并行（Pipeline Parallelism, PP）的通信解耦**。在训练和部署百亿及千亿级模型（如 Qwen2.5-72B、Llama-3.1-405B、DeepSeek-V3）时，模型往往需要切分到多个节点上执行流水线并行：第 0 号 GPU 舞台（Stage 0）负责执行 embed\_tokens 和前几层 Decoder，而最后一个 GPU 舞台（Stage P-1）负责执行最后几层 Decoder 和 lm\_head。如果启用了 tie\_word\_embeddings: true，Stage 0 和 Stage P-1 上就必须各自维护一份相同的 \[V, d\] 权重副本，并且在每一个训练 Step 的反向传播结束时，必须在首尾两个物理距离最远的节点之间发起一次昂贵的跨节点点对点 All-Reduce 梯度同步以确保权重一致（正如 MAX 等现代推理框架在实现 DeepSeek-V3 等大模型图编译时，甚至直接抛出异常拒绝支持 tie\_word\_embeddings）。将首尾权重彻底解绑（tie\_word\_embeddings: false），让 embed\_tokens 安分地留在首节点、让 lm\_head 独立地留在尾节点，不仅消除了跨流水线首尾的通信同步开销，更让现代大模型的分布式架构设计变得前所未有地纯粹与高效。

## **系列博客的课程设计总结与架构演进脉络收敛**

纵观从 2017 年经典 Transformer / GPT-2 到现代 Llama-3 与 Qwen2.5 的这六处核心架构演进，可以清晰地提炼出一条贯穿现代大语言模型演进的底层逻辑：**在不改动 \[B, T\] \-\> \[B, T, d\] \-\> \[B, T, V\] 这一宏观张量流水线契约的前提下，将一切阻碍训练规模化与推理低延迟的微观算子全部重塑**。为了让网络能够稳定堆叠至数十乃至上百层并跑在低精度浮点数上，架构师用 **Pre-RMSNorm** 清空了残差主干上的路障，并移除了易引发数值漂移的 **Linear Bias**（或在 Qwen3 中进一步用 **QK-Norm** 替代了早期为辅助外推而保留的 **QKV-Bias**）。为了让模型突破固定上下文长度的枷锁，架构师废弃了入口处臃肿的绝对位置矩阵，代之以每层注意力内部在四维拆分头张量 \[B, h, T, d\_h\] 上动态施加的 **RoPE 旋转位置编码**。为了打破自回归生成时的显存带宽墙，架构师通过 **GQA** 将键值投影矩阵收缩为原来的 1/g（\[h\_kv \* d\_h, d\]），并将省下的参数与算力预算投入到表达能力更强的 **三矩阵 SwiGLU 前馈网络** 与 **解绑的独立超大词表投影层** 之中。  
在撰写面向开发者的技术系列博客时，坚守“《第零篇》锁定三维与四维张量形状流转、《第 2 篇》深挖数学恒等变换与硬件 Roofline 瓶颈”的双层叙事结构，不仅能帮助读者在几分钟内无障碍地读懂任何主流开源模型的 PyTorch 前向推理代码，更能使其在后续深入算子融合、KV Cache 显存管理与分布式并行时，建立起坚实的系统级第一性原理认知。

#### **引用的文献**

1\. Read a Hugging Face model config \- MAX, https\://max.modular.com/develop/read-model-config/ 2\. qwen2 \- vLLM docs, https\://docs.vllm.ai/en/v0.11.2/api/vllm/model\_executor/models/qwen2.html 3\. modeling\_qwen.py · Alibaba-NLP/gte-Qwen2-7B-instruct at main, https\://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/blob/main/modeling\_qwen.py 4\. Qwen3 Technical Report \- arXiv, https\://arxiv.org/html/2505.09388v1 5\. Qwen2 Transformer Architecture \- Emergent Mind, https\://www\.emergentmind.com/topics/qwen2-transformer-architecture 6\. Qwen2MoE \- Hugging Face, https\://huggingface.co/docs/transformers/model\_doc/qwen2\_moe 7\. config.json · Qwen/Qwen2.5-32B at main \- Hugging Face, https\://huggingface.co/Qwen/Qwen2.5-32B/blob/main/config.json 8\. Download config.json from Qwen/Qwen2.5-0.5B \- Hugging Face, https\://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json 9\. config.json · Qwen/Qwen2.5-14B at main \- Hugging Face, https\://huggingface.co/Qwen/Qwen2.5-14B/blob/main/config.json 10\. config.json · Qwen/Qwen2.5-VL-7B-Instruct at main \- Hugging Face, https\://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json 11\. config.json · Qwen/Qwen2.5-7B-Instruct at main \- Hugging Face, https\://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json 12\. config.json · Qwen/Qwen2.5-7B at main \- Hugging Face, https\://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json 13\. config.json · Qwen/Qwen2.5-1.5B-Instruct at main \- Hugging Face, https\://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json 14\. Can we change mistral attention bias=False to bias=config ... \- GitHub, https\://github.com/huggingface/transformers/issues/29553 15\. Qwen2.5-1.5B 进行SFT训练 grad\_norm固定为1.73205 loss一开始很, https\://github.com/hiyouga/LlamaFactory/issues/9567 16\. configuration\_audio8\_tts \- vLLM-Omni, https\://docs.vllm.ai/projects/vllm-omni/en/v0.30.0/api/vllm\_omni/model\_executor/models/audio8\_tts/configuration\_audio8\_tts/ 17\. Qwen Models: Evolution of Multimodal LLMs \- Emergent Mind, https\://www\.emergentmind.com/topics/qwen-models 18\. Writing an LLM from scratch, part 32d \-- Interventions: adding, https\://www\.gilesthomas.com/2026/02/llm-from-scratch-32d-interventions-adding-attention-bias 19\. Understanding Qwen-v1: My Personal Take \| by tangbasky \- Medium, https\://medium.com/data-science-collective/understanding-qwen-v1-my-personal-take-f7b302111d31 20\. Qwen Architecture: Alibaba's Multilingual LLM Design \- Interactive, https\://mbrenndoerfer.com/writing/qwen-architecture-multilingual-llm 21\. config.json · second-state/Meta-Llama-3.1-8B-Instruct-GGUF at, https\://huggingface.co/second-state/Meta-Llama-3.1-8B-Instruct-GGUF/blame/112ac8d00f6685451451ea8ac393168c3f9042d0/config.json 22\. config.json · Qwen/Qwen2.5-0.5B-Instruct at main \- Hugging Face, https\://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blame/main/config.json 23\. config.json · Qwen/Qwen2.5-1.5B at refs/pr/9 \- Hugging Face, https\://huggingface.co/Qwen/Qwen2.5-1.5B/blob/refs%2Fpr%2F9/config.json 24\. config.json · mlx-community/Meta-Llama-3.1-405B-Instruct-8bit at main, https\://huggingface.co/mlx-community/Meta-Llama-3.1-405B-Instruct-8bit/blob/main/config.json 25\. ali-vilab/Unison \- GitHub, https\://github.com/ali-vilab/Unison 26\. Taming the LLaMA: A Guide to Herding LLaMA 3.1 with Hugging Face, https\://snawarhussain.com/educational/llms/Taming-the-LLaMA-3.1/