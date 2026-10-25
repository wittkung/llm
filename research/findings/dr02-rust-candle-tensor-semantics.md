# **基于 Rust Candle 框架的白盒大语言模型前向计算追踪器：底层张量布局、算子语义与数值对齐深度研究报告**

## **架构背景与白盒追踪的核心挑战**

在构建教学级与诊断级的大语言模型（LLM）白盒前向计算追踪器时，工程实现的核心诉求在于剥离高层黑盒模型封装（如直接调用 candle-transformers 中的端到端模型结构），转而使用 candle-core 与 candle-nn 的基础张量原语显式编写每一层线性投影、位置编码、分组查询注意力（GQA）及激活函数，并在算子边界插入张量形状（Shape）、步长（Stride）与数值切片探针。  
与 PyTorch 等动态图框架不同，Hugging Face 的 candle 框架（当前主流版本 0.11.0）采用了极简的不可变张量视图与显式内存布局（Layout）设计。在 candle 中，张量由底层物理存储（Storage，通过 Arc 引用计数实现廉价克隆）与逻辑视图元数据（Layout，包含维度 Shape、各维度步长 Stride 以及起始偏移量 start\_offset）共同定义。这种设计虽然赋予了零拷贝视图变换极高的执行效率，但也意味着算子对内存连续性（Contiguity）、批次维度广播（Batch Broadcasting）、旋转位置编码（RoPE）的内存配对约定以及硬件后端的数据类型（DType）具有极其严格的底层约束。任何一处维度步长或配对方式的隐式错位，轻则在运行期触发 Error::RequiresContiguous 或 Error::ShapeMismatchBinaryOp 异常，重则在深层网络中引发跨层累积的静默数值漂移。

## **Safetensors 权重存储方向与矩阵乘法语义解析**

### **权重存储方向与转置视图的内存语义**

在 PyTorch 生态中，nn.Linear(in\_features, out\_features) 模块内部维护的权重参数形状固定为 \[out\_features, in\_features\]，这一约定被原样保留在导出的 .safetensors 权重文件中。在 candle-nn 的源码实现（candle-nn/src/linear.rs）中，Linear 结构体同样直接持有形状为 (out\_dim, in\_dim) 的二维权重张量 weight 以及可选的一维偏置张量 bias（形状为 out\_dim），并通过 VarBuilder 按字符串键名 "weight" 与 "bias" 从映射文件中提取张量：  
`pub struct Linear {`  
    `weight: Tensor,`  
    `bias: Option<Tensor>,`  
`}`

`pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> candle_core::Result<Linear> {`  
    `let weight = vb.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;`  
    `let bias = vb.get(out_dim, "bias")?;`  
    `Ok(Linear::new(weight, Some(bias)))`  
`}`

当输入激活张量 X 的形状为三维 \[B, T, d\_in\]（分别对应批次大小 B、序列长度 T 与输入特征维度 d\_in）时，前向传播的数学表达为 Y \= X W^\\top \+ b。在 candle-core 中，对形状为 \[d\_out, d\_in\]、物理步长为 \[d\_in, 1\] 的连续权重张量调用 .t()?（即 .transpose(D::Minus2, D::Minus1)?）时，并不会发生任何物理内存拷贝，而是返回一个形状为 \[d\_in, d\_out\]、步长为 \[1, d\_in\] 的非连续转置视图。

### **三维激活与二维转置权重直接调用 matmul 的报错机制**

在白盒追踪器中，若直接对三维张量 x（\[B, T, d\_in\]）与二维转置权重 w.t()?（\[d\_in, d\_out\]）调用 x.matmul(\&w.t()?)，**必然会在运行期报错并抛出 Error::ShapeMismatchBinaryOp 异常**。  
审查 candle-core/src/tensor.rs 中 Tensor::matmul 的底层实现可以发现，matmul 被严格定义为**同秩批次矩阵乘法（Same-Rank Batched Matrix Multiplication）**，不支持不同秩（Rank）张量之间的隐式维度广播，也不支持批次维度的自动扩展。其内部校验逻辑如下：  
`let batching_b: usize = b_dims[..dim - 2].iter().product();`  
`if k != k2[span_6](start_span)[span_6](end_span)[span_8](start_span)[span_8](end_span) || batching != batching_b {`  
    `Err(Error::ShapeMismatchBinaryOp {`  
        `lhs: self.shape().clone(),`  
        `rhs: rhs.shape().clone(),`  
        `op: "matmul",`  
    `}`  
    `.bt())?`  
`}`

在执行过程中，Tensor::matmul 首先进行秩对齐校验，要求左操作数 lhs 与右操作数 rhs 的维度数量（Rank）完全一致且均不小于 2；当 x 为三维张量（Rank \= 3）而 w.t()? 为二维张量（Rank \= 2）时，秩校验直接失败。即便两个操作数同秩，记总维度数为 dim，matmul 也会提取前 dim \- 2 个维度的乘积作为批次总数 batching（对于 \[B, T, d\_in\]，batching \= B \* T），并严格要求左右操作数的前置批次维度乘积（batching \== batching\_b）及内积收缩维（k \== k2）完全相等，校验通过后方可将 (batching, m, n, k) 连同两者的 Layout 传递给底层 storage().matmul(...)。

### **broadcast\_matmul 与 Linear::forward 的底层处理机制对比**

为了解决三维（或更高维）激活张量 \[B, T, d\_in\] 与二维权重 \[d\_in, d\_out\] 的乘法问题，candle 提供了两条技术路径：candle\_core::Tensor::broadcast\_matmul 与 candle\_nn::Linear::forward 的内部重塑与广播机制。  
在 candle-core/src/tensor.rs 中，broadcast\_matmul 的源码实现如下：  
`pu[span_26](start_span)[span_26](end_span)b fn broadcast_matmul(&self, rhs: &Self) -> Result<Self> {`  
    `let lhs = self;`  
    `let (l_shape, r_shape) = lhs.shape().broadcast_shape_matmul(rhs.shape())?;`  
    `let l_broadcast = l_shape != *lhs.shape();`  
    `let r_broadcast = r_shape != *rhs.shape();`  
    `// TODO: Avoid concretising the broadcasted matrixes via contiguous.`  
    `match (l_broadcast, r_broadcast) {`  
        `(true, true) => lhs`  
            `.broadcast_as(&l_shape)?`  
            `.contiguous()?`  
            `.matmul(&rhs.broadcast_as(&r_shape)?.contiguous()?),`  
        `(false, true) => lhs.matmul(&rhs.broadcast_as(&r_shape)?.contiguous()?),`  
        `(true, false) => lhs.broadcast_as(&l_shape)?.contiguous()?.matmul(rhs),`  
        `(false, false) => lhs.matmul(rhs),`  
    `}`  
`}`

分析上述源码可以揭示一个极为关键的工程细节：当 lhs 为 \[B, T, d\_in\]、rhs 为 w.t()?（\[d\_in, d\_out\]）时，broadcast\_shape\_matmul 计算出的目标形状分别为 l\_shape \= \[B, T, d\_in\] 和 r\_shape \= \[B, d\_in, d\_out\]。此时匹配到 (fals\[span\_27\](start\_span)\[span\_27\](end\_span)e, true) 分支，broadcast\_matmul 会对 rhs 调用 .broadcast\_as(\&r\_shape)?.contiguous()?。如源码注释 TODO:\[span\_28\](start\_span)\[span\_28\](end\_span) Avoid concretising the broadcasted matrixes via contiguous 所示，这一调用会产生显著的物理内存具象化（Materialization）开销，将原本通过 .t()? 零拷贝转置并广播为 \[B, d\_in, d\_out\] 的权重矩阵，在物理内存中真正转置并复制 B 份，然后调用三维 matmul。此外，broadcas\[span\_29\](start\_span)\[span\_29\](end\_span)t\_matmul 之所以强制调用 .contiguous()?，还与底层后端的零步长批次（Stride-Zero Batch）缺陷直接相关：根据 candle 上游 Issue \#3253 与 Issue \#3744 的验证报告，在 candle 0.11.0 中，如果直接对 broadcast\_as 产生的零步长批次张量调用 .matmul()（而不经过 .contiguous()），未开启 accelerate 的基础 CPU 后端（基于 gemm crate）会静默计算出错误的数值，只有开启 Apple Accelerate 后端时才能正确处理零步长批次布局。  
相比之下，candle\_nn::Linear::forward（位于 candle-nn/src/linear.rs）与 candle-transformers 中的模型实现采用了更为高效的二维重塑快路径（Reshape-to-2D Fast Path）。当输入 x 为三维 \[bsize, seq\_len, in\_dim\] 且内存连续时（或者在 quantized\_llama.rs 等模型实现中通过 xs.reshape(((), hidden\_dim))? 显式展平），框架会将 \[B, T, d\_in\] 通过零拷贝 .reshape((B \* T, d\_in))? 展平为二维矩阵 \[B \* T, d\_in\]，直接与二维转置视图 self.weight.t()?（\[d\_in, d\_out\]）执行标准二维 matmul，得到 \[B \* T, d\_out\] 后再通过 .reshape((B, T, d\_out))? 恢复为三维张量，最后通过 .broadcast\_add(\&bias)? 叠加偏置。这一二维重塑路径不仅彻底消除了 broadcast\_matmul 中 .contiguous()? 对转置权重的物理内存拷贝，而且在底层 GEMM 调用中直接利用了 BLAS 的转置标志位读取原始 \[d\_out, d\_in\] 连续内存，将 B 次小规模矩阵乘法合并为单次大规模矩阵乘法（M \= B \* T, N \= d\_out, K \= d\_in）。

| 矩阵乘法实现方式 | 代码调用形式（输入 x: \[B, T, d\_in\], 权重 w: \[d\_out, d\_in\]） | 是否报错 | 权重内存拷贝行为 | 适用场景与白盒追踪建议 |
| :---- | :---- | :---- | :---- | :---- |
| 直接调用 matmul | x.matmul(\&w.t()?) | **报错** (ShapeMismatchBinaryOp) | 无（在形状校验阶段即终止） | 严禁对不同秩张量直接调用 |
| 广播矩阵乘法 broadcast\_matmul | x.broadcas\[span\_30\](start\_span)\[span\_30\](end\_span)t\_matmul(\&w.t()?\[span\_31\](start\_span)\[span\_31\](end\_span)) | 正常执行 | 触发 .contiguous()，物理转置并复制 B 份权重 | 语法最简；B=1 时开销可控，但 B\>1 时存在冗余内存分配 |
| 零步长手动广播 broadcast\_as \+ matmul | x.matmul(\&w.t()?.broadcast\_as((b,\[span\_35\](start\_span)\[span\_35\](end\_span) d\_in, d\_out))?) | 基础 CPU 后端**静默算错**或拒绝 | 零拷贝（批次步长为 0） | **高危陷阱**（Issue \#3744），严禁在未开启 acce\[span\_39\](start\_span)\[span\_39\](end\_span)lerate 的 CPU 上使用 |
| 2D 重塑乘法（推荐白盒范式） | x.reshape((b \* t, d\_in))?.ma\[span\_41\](start\_span)\[span\_41\](end\_span)tmul(\&w.t()?)?.reshape((b, t, d\_out))? | 正常执行 | **零拷贝**（利用 GEMM 原生转置步长） | **最佳实践**：数学语义清晰、无权重拷贝、全后端数值绝对安全 |

因此，在编写白盒追踪器的显式线性投影函数时，最严谨的实现方式是封装一个辅助函数（或直接调用 candle\_nn::Linear::forward），通过 reshape 将前缀维度合并为二维执行 matmul(\&w.t()?)，再恢复至 \[B, T, d\_out\]。

## **多头拆分与张量内存连续性（Contiguity）边界分析**

### **转置操作引发的非连续步长（Non-Contiguous Stride）演变**

在多头注意力（MHA）或分组查询注意力（GQA）中，线性投影输出的张量形状为 \[B, T, h \* d\_h\]，其在物理内存中按行优先（Row-Major / C-Contiguous）紧密排列，步长为 \[T \* h \* d\_h, h \* d\_h, 1\]。当执行多头拆分与维度置换时，首先调用 .reshape((b\_sz, seq\_len, n\_head, head\_dim))? 进行四维重塑，张量逻辑形状变为 \[B, T, h, d\_h\]，步长变为 \[T \* h \* d\_h, h \* d\_h, d\_h, 1\]。由于这一步只是对最内层连续维度进行因式分解，张量依然保持连续（is\_contiguous() \== true）。然而，紧接着调用 .t\[span\_50\](start\_span)\[span\_50\](end\_span)ranspose(1, 2)? 交换序列维与头维后，逻辑形状变为 \[B, h, T, d\_h\]。因为 .transpose(1, 2\) 仅交换了 Layout 中的维度大小与对应步长，导致步长变为 \[T \* h \* d\_h, \[span\_51\](start\_span)\[span\_51\](end\_span)d\_h, h \* d\_h, 1\]。此时，沿第 1 维（头维 h）移动一个元素的物理跨度为 d\_h，而沿第 2 维（序列维 T）移动一个元素的物理跨度为 h \* d\_h。由于步长不再满足从外向内单调递减的紧凑乘积关系，张量在物理内存中变为**非连续状态（is\_contiguous() \== false）**。

### **RoPE 旋转位置编码与 V 张量的连续性强制要求**

在得到非连续的 \[B, h, T, d\_h\] 张量（对于 Q 为 \[B, h\_q, T, d\_h\]，对于 K, V 为 \[B, h\_kv, T, d\_h\]）之后，candle 中后续各个算子对内存连续性的要求存在本质差异。在 candle-nn 中，rotary\_emb::{rope, rope\_i, rope\_thd} 是通过内部自定义算子接口 apply\_op3\_no\_bwd 注册的底层原语。其 CPU、CUDA 与 Metal 内核均直接通过 layout.contiguous\_offsets() 获取底层一维连续切片指针，并按固定偏移量计算 (b, h, t, d) 下标。如果直接将 .transpose(1, 2)? 后的非连续张量传入 rope 或 rope\_i，candle-core 会立即返回 Error::RequiresContiguous。因此，在调用 candle\_nn::rotary\_emb::rope(\&q.contiguous()?, \&cos, \&sin)? 与 rope(\&k.contiguous()?, \&cos, \&sin)? 时，必须对输入显式调用 .contiguous()?。值得注意的是，由于 rope 算子会分配一块全新的物理内存来存放旋转后的计算结果，其返回的 q 和 k 张量本身已经是内存连续的 \[B, h, T, d\_h\] 张量。  
相比之下，由于价值张量 v 不参与 RoPE 运算，它不会像 q 和 k 那样在 RoPE 阶段被连带转换为连续张量。正如 quantized\_llama.rs 第 209–215 行源码及注释所明确指出的，必须在对 v 执行 .transpose(1, 2)? 后显式调用 .contiguous()?：  
`let q = q`  
    `.re[span_52](start_span)[span_52](end_span)shape((b_sz, seq_len, self.n_head, self.head_dim))?`  
    `.transpose(1, 2)?;`  
`let k = k`  
    `.reshape((b_s[span_53](start_span)[span_53](end_span)z, seq_len, self.n_kv_head, self.head_dim))?`  
    `.transpose(1, 2)?;`  
`let v = v`  
    `.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?`  
    `.transpose(1, 2)?`  
    `// This call to contiguous ensures that the fast kernel can be called below. It's`  
    `// actually a no-op except when processing the initial prompt so has no significant`  
    `// impact on performance.`  
    `.contiguous()?;`

当自回归解码阶段 seq\_len \== 1 时，由于形状 \[B, 1, h, d\_h\] 与 \[B, h, 1, d\_h\] 中长度为 1 的维度不影响元素排列顺序，is\_contiguous() 仍为 true，此时 .contiguous()? 是一个零开销的空操作（No-Op）；但在处理初始 Prompt（即 seq\_len \> 1）时，该调用是必不可少的物理重排操作。

### **GQA 键值扩展、注意力矩阵乘法与多头合并的连续性机制**

在 GQA 键值扩展（repeat\_kv）阶段，当采用 unsqueeze(2)?.expand(...)?.reshape(...)（或 candle\_transformers\[span\_54\](start\_span)\[span\_54\](end\_span)::utils::repeat\_kv）将 k 和 v 从 \[B, h\_kv, T, d\_h\] 扩展至 \[B, h\_q, T, d\_h\] 时，.expand(...) 会产生第 2 维步长为 0 的非连续广播视图。但在紧随其后的 .reshape((b\_sz, n\_head, seq\_len, head\_dim))? 调用中，candle-core 的 Tensor::reshape\[span\_55\](start\_span)\[span\_55\](end\_span) 会在检测到输入非连续时自动在内部调用 .contiguous()? 完成物理内存具象化，因此 repeat\_kv 的输出张量必然是连续的。然而，若在白盒代码中仅使用 .expand() 或 .broadcast\_as() 而不调用 .reshape() 或 .contiguous()，直接将其送入 matmul 则会在基础 CPU 后端触发零步长批次计算错误（Issue \#3744）。  
在随后的注意力分数矩阵乘法 q.matmul(\&k.t()?) 中，经过 RoPE 后连续的 q（形状 \[B, h\_q, T, d\_h\]）与 k.t()?（对连续的 k 交换最后两个维度，形状 \[B, h\_q, d\_h, T\]，步长 \[h\_q\*T\*d\_h, T\*d\_h, 1, d\_h\]）直接调用 q.matmul(\&k.t()?) **完全合法且无需调用 .contiguous()?**。这是因为 can\[span\_42\](start\_span)\[span\_42\](end\_span)dle-core 的 matmul 布局检查器专门支持最后两个维度的转置步长：只要张量的前置批次维度（B 和 h\_q）步长连续，且最后两个维度中有一个维度的步长为 1（此处 k.t()? 的倒数第二维步长为 1，最后一维步长为 d\_h），底层 GEMM（无论是 CPU gemm、Apple Accelerate 还是 Metal/CUDA）都会直接通过设置矩阵转置标志位原位读取 k 的内存，从而省去一次昂贵的物理转置拷贝。  
与之形成鲜明对比的是，在计算注意力加权求和 att.matmul(\&v) 时，v 必须保持连续内存布局。在 quantized\_lla\[span\_75\](start\_span)\[span\_75\](end\_span)\[span\_80\](start\_span)\[span\_80\](end\_span)ma.rs 第 258–259 行可以看到如下关键源码与注释：  
`let att = candle_nn::ops::softmax_last_dim(&att)?;`  
`// Convert to contiguous as matmu[span_57](start_span)[span_57](end_span)l doesn't support strided vs for now.`

为什么 \[span\_58\](start\_span)\[span\_58\](end\_span)k.t()? 可以非连续，而 v 在 att.matmul(\&v) 中必须连续？因为 v 在最初多头拆分时执行的是 .transpose(1, 2)?（交换的是第 1 维 h 与第 2 维 T），其最后两个维度 \[T, d\_h\] 的步长变成了 \[h \* d\_h, 1\]。这种跨头交错的步长（主维跨度 lda \= h \* d\_h \> d\_h\[span\_59\](start\_span)\[span\_59\](end\_span) 且批次维度步长倒置）不属于标准二维矩阵转置，底层批次 GEMM 拒绝接受此类跨批次交错步长的右操作数 v。因此，v 必须在 .transpose(1, 2)? 后（或通过 repeat\_kv）转换为连续内存。  
最后，在多头合并阶段，当 att.matmul(\&v) 输出形状为 \[B, h\_q, T, d\_h\] 的连续张量 y 后，调用 y.transpose(\[span\_60\](start\_span)\[span\_60\](end\_span)1, 2)? 将其还原为 \[B, T, h\_q, d\_h\] 时，张量再次变为非连续状态。与 PyTorch 中对非连续张量调用 .view() 会报错不同，candle-core 没有区分 view 与 reshape，其 Tensor::reshape 方法内部包含自动连续化逻辑：当检测到 \!self.is\_contiguous() 时，会自动先执行 self.contiguous()? 再重塑形状。这意味着直接编写 y.transpose(1, 2)?.reshape((b\_sz, seq\_len, hidden\_size))? 不会在运行时报错，但在教学级白盒追踪器中，为了显式展示物理内存重排发生的时机并方便插入探针，推荐显式写作 y.transpose(1, 2)?.contiguous()?.reshape((b\_sz, seq\_len, hidden\_size))?。

## **RoPE（旋转位置编码）的两种内存配对流派与模型对齐**

### **两种配对流派的数学与内存布局本质**

旋转位置编码（RoPE）通过在头维度 d\_h 上将特征向量划分为 d\_h / 2 个二维复数平面子空间，对每一对分量施加旋转角 m \\theta\_i（其中 m 为序列位置下标，\\theta\_i \= \\text{base}^{-2i / d\_h}，i \\in \\{0, 1, \\dots, d\_h/2 \- 1\\}）。在工程实现中，这 d\_h / 2 个二维子空间的下标配对存在两种截然不同的流派。  
第一种流派是 **GPT-NeoX / HuggingFace rotate\_h\[span\_87\](start\_span)\[span\_87\](end\_span)\[span\_96\](start\_span)\[span\_96\](end\_span)alf 风格（非交错半区配对，Non-Interleaved）**，它将头维度 d\_h 从中间一分为二，前半截下标 i \\in \[0, d\_h/2) 与后半截下标 i \+ d\_h/2 \\in \[d\_h/2, d\_h) 组成二维旋转对 (x\_i, x\_{i \+ d\_h/2})。其单步旋转公式为：  
\\begin{pmatrix} x'\_i \\\\ x'\_{i \[span\_88\](start\_span)\[span\_88\](end\_span)\[span\_97\](start\_span)\[span\_97\](end\_span)+ d\_h/2} \\end{pmatrix} \= \\begin{pmatrix} \\cos(m\\theta\_i) & \-\\sin(m\\theta\_i) \\\\ \\sin(m\\theta\_i) & \\cos(m\\theta\_i) \\end{pmatrix} \\begin{pmatrix} x\_i \\\\ x\_{i \+ d\_h/2} \\end{pmatrix}  
在 candle-nn 中，这一非交错半区配对流派对应函数 **candle\_nn::rotary\_emb::rope(\&x.contiguous()?, \&cos, \&sin)**。  
第二种流派是 原始 Meta LLaMA / llama.cpp NORM 风格（相邻奇偶交错配对，Interleaved），它将相邻的偶数下标 2i 与奇数下标 2i+1 组成二维旋转对 (x\_{2i}, x\_{2i+1})。其单步旋转公式为：  
\\begin{pmatrix} x'\_{2i} \\\\ x'\_{2i+1} \\end{pmatrix} \= \\begin{pmatrix} \\cos(m\\theta\_i) & \-\\sin(m\\theta\_i) \\\\ \\sin(m\\theta\_i) & \\cos(m\\theta\_i) \\end{pmatrix} \\begin{pmatrix} x\_{2i} \\\\ x\_{2i+1} \\end{pmatrix}  
在 candle-nn 中，这一奇偶交错配对流派对应函数 **candle\_nn::rotary\_emb::rope\_i(\&x.contiguous()?, \&cos, \&sin)**（其中后缀 \_i 即代表 Interleaved）。

### **SmolLM2、Llama-3.2 与 Qwen2.5 在 Safetensors 与 GGUF 中的配对真相**

在构建白盒追踪器时，一个极易引发困惑的问题是：既然 Meta 原始 LLaMA 采用的是奇偶交错配对（流派 b），那么从 Hugging Face Hub 下载的 Llama-3.2、SmolLM2 和 Qwen2.5 的 **.safetensors 权重文件**究竟对应哪一种流派？  
答案非常明确：**在 Hugging Face 官方 .safetensors 权重格式下，SmolLM2、Llama-3.2 与 Qwen2.5 全部统一对应流派 (a)（即 GPT-NeoX 非交错半区配对 rotate\_half），在 candle 中必须统一调用 candle\_nn::rotary\_emb::rope，严禁调用 rope\_i**。  
理解这一结论需要厘清 Hugging Face transformers 与 GGUF（llama.cpp）在权重转换链路上的历史演进与行重排机制。当 Meta 发布原始 LLaMA 权重时，其参考代码使用复数视图对相邻奇偶维度 (2i, 2i+1) 进行旋转。然而，Hugging Face 在将 LLaMA 集成进 transformers 库（modeling\_llama.py）时，直接复用了 GPT-NeoX 的 rotate\_half 代码（即半区配对 (i, i \+ d\_h/2)）。为了使模型前向输出与 Meta 原始代码在数学上完全等价，Hugging Face 的官方权重转换脚本（convert\_llama\_weights\_to\_hf.py）在生成 .safetensors 文件时，预先对每一层 W\_Q 和 W\_K 权重矩阵的输出行（Rows）执行了置换（Permutation），将原本散落在偶数位 2i 的特征行集中排列到前半区 \[0 .. d\_h/2)，将奇数位 2i+1 的特征行集中排列到后半区 \[d\_h/2 .. d\_h)。因此，Hugging Face Hub 上的 Llama-3.2 .safetensors 权重已全部按 HF LlamaForCausalLM 标准完成行置换，原生匹配 rotary\_emb::rope；SmolLM2（如 SmolLM2-135M / 360M / 1.7B）由 Hugging Face 团队基于 LlamaForCausalLM 架构从头预训练，训练期即直接使用 rotate\_half，权重天然为半区配对；而 Qwen2.5 采用 Qwen2ForCausalLM 架构，从预训练起同样原生采用 GPT-NeoX 风格的半区配对 rotate\_half。  
之所以在社区讨论中频繁出现 rope 与 rope\_i 的混淆，源于 GGUF 格式的逆向重排分叉（详见 candle Issue \#3410）。在 llama.c\[span\_112\](start\_span)\[span\_112\](end\_span)pp 生态中，convert\_hf\_to\_gguf.py 脚本在将 Hugging Face 的 LlamaForCausalLM（含 Llama-3.2）转换为 .gguf 格式时，会对 W\_Q 和 W\_K 执行逆向行置换（Un-permute），将其还原为 Meta 原始的奇偶交错顺序（标记为 rope\_type \= 0 / NORM），因此在加载 GGUF 的 quantized\_llama.rs 中，标准 Llama 模型需要调用 rope\_i；而 convert\[span\_113\](start\_span)\[span\_113\](end\_span)\_hf\_to\_gguf.py 在转换 Qwen2、Phi-3、Falcon、OLMo2 等原生 NEOX 架构时**不执行逆置换**（标记为 rope\_type \= 2 / NEOX），因此即便在 GGUF 中，Qwen2 也必须调用 rope。  
在 candl\[span\_114\](start\_span)\[span\_114\](end\_span)e Issue \#3410 的白盒探针实测中，研究人员通过转储第 0 层（Layer 0）中间张量揭示了误用 rope\_i 对 NEOX 风格权重造成的灾难性影响：在 Layer 0 中，attn\_norm（RmsNorm 输出）与参考实现达到逐位完全一致（误差为 0.00000），Q/K/V 线性投影加偏置的数值误差也严格控制在 0.00004 以内；然而一旦进入 RoPE 并计算注意力输出 kqv\_out（在乘 W\_O 之前），误用 rope\_i 会导致张量数值立即发生严重发散，最大绝对误差飙升至 0.53，且误差幅度随不同注意力头的频率段分布而变化。这一层内误差在经过 48 层 Transformer 块的指数级（Softmax）放大与累积后，产生系统性的方向偏置，导致模型将 \<\|endoftext\|\> 特殊控制 Token 的 Logit 异常虚高 \+11.7，使生成文本在约 30 个 Token 后陷入死循环重复。当修复为调用 ca\[span\_116\](start\_span)\[span\_116\](end\_span)ndle\_nn::rotary\_emb::rope 后，经过全部 48 层及 15.2 万词表投影后的最终 Logit 误差完全收敛至 \< 0.01 级别，前 10 名候选 Token 排序完全一致。

| 验证维度 / 目标 Token | 词汇表 Token ID | Candle 修复后 (rotary\_emb::rope) | 参考基准 (llama.cpp NEOX) | 绝对误差 (\\Delta) | 错误调用 rope\_i 时的异常表现 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Layer 0 attn\_norm 探针** | — | 基准对齐 | 基准对齐 | 0.00000 (逐位一致) | 尚未进入 RoPE，无误差 |
| **Layer 0 Q/K/V 投影探针** | — | 基准对齐 | 基准对齐 | 0.00004 | 尚未进入 RoPE，无误差 |
| **Layer 0 kqv\_out 注意力探针** | — | 基准对齐 | 基准对齐 | \< 0.0001 | **严重发散（最大绝对误差 0.53）** |
| **48 层最终 Logit: "I"** | 40 | 19.868 | 19.866 | 0.002 | 自然语言 Token Logit 被系统性压低 |
| **48 层最终 Logit: "Let"** | 10061 | 15.340 | 15.347 | 0.007 | 自然语言 Token Logit 被系统性压低 |
| **48 层最终 Logit: endoftext** | 151643 | 7.074 | 7.073 | 0.001 | **Logit 异常膨胀 \+11.7（生成 \~30 词后崩溃）** |
| **48 层最终 Logit: EOS** | 151645 | 4.481 | 4.481 | 0.000 | 特殊控制 Token 概率分布失真 |

### **与 HuggingFace Python 逐位对齐（误差 \< 1e-5）的 cos/sin 形状细节**

在调用 candle\_nn::rotary\_emb::rope 与 HuggingFace Python 进行逐位对齐时，除了选对 rope 函数外，还必须注意 **cos 与 sin 预计算张量的最后一维长度差异**。在 HuggingFace Python 的 modeling\_llama.py 与 modeling\_qwen2.py 中，为了配合手写的 rotate\_half(x) 广播乘法，Python 代码会执行 emb \= torch.cat((freqs, freqs), dim=-1)，使 cos 和 sin 的形状变为 **\[T, d\_h\]**（最后一维等于完整的 head\_dim）。而在 candle\_nn::rotary\_emb::rope(\&x.contiguous()?, \&cos, \&sin) 中，底层算子内核直接按半区下标 i \\in \[0, d\_h/2) 索引旋转角，因此**要求传入的 cos 和 sin 形状必须为 \[T, d\_h / 2\]**（即不执行 cat((freqs, freqs)) 拼接，直接保留 d\_h / 2 个频率的余弦和正弦值，且 cos、sin 必须为连续内存并与 x 的 DType 一致）：  
`[span_140](start_span)[span_140](end_span)// 构造与 candle_nn::rotary_emb::rope 完美对齐的 cos/sin 缓存 (形状: [max_seq_len, head_dim / 2])`  
`let inv_freq: Vec<f32> = (0..head_dim)`  
    `.step_by(2)`  
    `.map(|i| 1f32 / rope_theta.powf(i as f32 / head_dim as f32))`  
    `.collect();`  
`let inv_freq_len = inv_freq.len(); // 等于 head_dim / 2`  
`let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), &device)?;`  
`let t = Tensor::arange(0u32, max_seq_len as u32, &device)?`  
    `.to_dtype(DType::F32)?`  
    `.reshape((max_seq_len, 1))?;`  
`let freqs = t.matmul(&inv_freq)?; // 形状: [max_seq_len, head_dim / 2]`  
`let cos = freqs.cos()?;`  
`let sin = freqs.sin()?;`

`// 前向传播中截取当前序列范围并执行 GPT-NeoX 风格半区配对 RoPE`  
`let cos = cos.narrow(0, index_pos, seq_len)?;`  
`let sin = sin.narrow(0, index_pos, seq_len)?;`  
`let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;`  
`let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;`

| 模型家族 | 权重格式 | 权重内部排列状态 | 旋转配对方式 | candle\_nn::rotary\_emb 对应函数 | cos / sin 形状要求 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **SmolLM2** | HF safetensors | 原生半区排列 | (i, i \+ d\_h/2) (NEOX) | rotary\_emb::rope | \[T, d\_h\[span\_141\](start\_span)\[span\_141\](end\_span) / 2\] |
| **Llama-3.2** | HF safetensors | 导出时已置换为半区排列 | (i, i \+ d\_h/2) (NEOX) | rotary\_emb::rope | \[T, d\_h\[span\_142\](start\_span)\[span\_142\](end\_span) / 2\] |
| **Qwen2.5** | HF safetensors | 原生半区排列 | (i, i \+ d\_h/2) (NEOX) | rotary\_emb::rope | \[T, d\_h\[span\_143\](start\_span)\[span\_143\](end\_span) / 2\] |
| **Llama-3.2** | GGUF (llama.cpp) | 转换时逆置换回奇偶交错 | (2i, 2i+1) (NORM) | rotary\_\[span\_144\](start\_span)\[span\_144\](end\_span)emb::rope\_i | \[T, d\_h\[span\_145\](start\_span)\[span\_145\](end\_span) / 2\] |
| **Qwen2.5** | GGUF (llama.cpp) | 保持原生半区排列 | (i, i \+ d\_h/2) (NEOX) | rotary\_\[span\_146\](start\_span)\[span\_146\](end\_span)emb::rope | \[T, d\_h\[span\_147\](start\_span)\[span\_147\](end\_span) / 2\] |

## **GQA（Grouped-Query Attention）的 KV 扩展实现范式**

在 SmolLM2、Llama-3.2 与 Qwen2.5 中，均广泛采用分组查询注意力（GQA）以压缩 KV Cache 显存占用。设查询头数为 h\_q，键值头数为 h\_{kv}（满足 h\_q \\pmod{h\_{kv}} \= 0），每组共享同一个 KV 头的查询头数量为 n\_{\\text{rep}} \= h\_q / h\_{kv}。此时 Q 的形状为 \[B, h\_q, T, d\_h\]，而 K, V 的形状为 \[B, h\_kv, T, d\_h\]。  
为了将 K, V 扩展为 \[B, h\_q, T, d\_h\]，同时严格保证**第 j 个 KV 头被连续复制 n\_{\\text{rep}} 次**（即扩展后的头下标 j \\cdot n\_{\\text{rep}} \+ r 对应原始第 j 个 KV 头，而非将整个 \[0..h\_kv\] 周期性交错平铺），在 candle-core 中最清晰、最地道的 4 行张量操作实现如下（与 candle\_transformers::utils::repeat\_kv 完全同源）：  
`pub fn repeat_kv(xs: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {`  
    `if n_rep == 1 {`  
        `Ok(xs)`  
    `} else {`  
        `let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;`  
        `xs.unsqueeze(2)?`  
            `.expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?`  
            `.reshape((b_sz, n_kv_head * n_rep, se[span_62](start_span)[span_62](end_span)q_len, head_dim))`  
    `}`  
`}`

上述实现之所以兼具数学清晰度与极佳的执行性能，源于其三步张量变换的精确配合。首先，调用 .unsqueeze(2)? 在 n\_kv\_head（第 1 维）紧后方插入一个长度为 1 的新广播轴（第 2 维），使张量形状从 \[B, h\_kv, T, d\_h\] 变为 \[B, h\_kv, 1, T, d\_h\]。这一插入位置至关重要：如果在维度 1（n\_kv\_head 之前）插入并重塑为 \[B, n\_rep \* h\_kv, T, d\_h\]，头的排列顺序会变成 \[0, 1, ..., h\_kv-1, 0, 1, ...\] 的周期性平铺，导致查询头与错误的 KV 头配对；只有插在 n\_kv\_head 之后（维度 2），合并后才能实现等价于 PyTorch torch.repeat\_interleave(dim=1, repeats=n\_rep) 的分组连续复制。其次，调用 .expand((b\_sz, n\_kv\_head, n\_rep, seq\_len, head\_dim))?（在 candle-core 中是 broadcast\_as 的别名方法）将第 2 维从 1 逻辑扩展为 n\_rep，并将该维度的步长设为 0，全程不分配任何新内存。最后，调用 .reshape((b\_sz, n\_kv\_head \* n\_rep, seq\_len, head\_dim)) 将第 1 维 n\_kv\_head 与第 2 维 n\_rep 合并为 h\_q \= n\_kv\_head \* n\_rep。由于此时的输入是包含零步长维度的非连续广播张量，reshape 内部会自动触发一次 .contiguous()?，在单次内存遍历中将 \[B, h\_q, T, d\_h\] 写入连续物理内存，使其可以直接安全地参与后续的 q.matmul(\&k.t()?) 与 att.matmul(\&v)。  
在追求极致内存带宽的生产级推理引擎（如 minillm）中，还可以通过将 Q 重塑为五维或折叠每个 KV 头对应的查询头组来直接执行矩阵乘法，从而避免物化扩展后的 K, V 副本（例如在 Qwen3-0.6B 的 16 查询头、8 KV 头配置下可节省 2 倍缓存空间）；但在需要将每一步中间张量与 HuggingFace Python 的 repeat\_kv 探针逐位比对的教学级白盒追踪器中，上述 unsqueeze(2)?.expand(...)?.reshape(...) 是完全对齐的标准范式。

## **数据类型（BF16 vs F32）与 macOS（CPU / Metal）兼容性策略**

现代开源大语言模型（包括 SmolLM2、Llama-3.2 与 Qwen2.5）的官方 .safetensors 权重文件几乎全部以 BF16（bfloat16）格式存储。在 macOS（Apple Silicon M1/M2/M3/M4 的 CPU 或 Metal 后端）上运行白盒追踪器时，**在加载 safetensors 阶段统一将权重转换为 DType::F32（.to\_dtype(DType::F32)?）不仅是推荐做法，更是保障 CPU 可执行性与跨层数值探针对齐的唯一最佳实践**。

### **CPU 后端硬性限制与 Metal 后端的探针精度要求**

在 candle-c\[span\_151\](start\_span)\[span\_151\](end\_span)\[span\_154\](start\_span)\[span\_154\](end\_span)ore（截至 0.11.0）的架构实现中，**CPU 后端根本没有实现针对 DType::BF16（或 F16）的矩阵乘法（GEMM）内核**。正如基于 candle 构建的推理引擎实测报告所指出的那样，candle 缺乏 CPU bf16 gemm 支持，因此在 CPU 上加载 bf16 检查点时必须回退转换为 fp32，否则一旦带着 BF16 类型进入 matmul 算子就会直接报错终止。  
虽然 Apple Silicon 的 Metal GPU 后端（candle-metal-kernels）支持 BF16 与 F16 算子执行，但在白盒追踪器场景下，即便使用 Metal 后端也应当统一采用 DType::F32。首先，从数值精度角度看，BF16 仅有 7 位尾数（精度约为十进制 2 \\sim 3 位有效数字，机器精度 \\epsilon \\approx 7.8 \\times 10^{-3}），而 F32 拥有 23 位尾数（\\epsilon \\approx 1.19 \\times 10^{-7}）。若要求白盒追踪器与 Python 参考实现达到 \< 1e-5 的逐位对齐，使用 BF16 会在第一层线性投影和 RmsNorm 处就产生 10^{-2} \\sim 10^{-3} 量级的舍入差异，彻底掩盖 \< 1e-5 精度的算子对齐验证。其次，从 Rust 强类型切片探针提取（to\_vec）的便利性来看，candle-core 中用于提取一维或二维 Rust Vec 以打印数值探针的方法 tensor.to\_vec1::\<T\>() 与 tensor.to\_vec2::\<T\>() 具有严格的运行时类型匹配检查。如果张量的运行时 DType 是 BF16，直接调用 .to\_vec1::\<f32\>() 会触发 Error::DTypeMismatch 报错——正如 quantized\_llama.rs 第 90 行在将 routing\_weights 提取为 Vec\<Vec\<f32\>\> 前必须先调用 routing\_weights.to\_dtype(DType::F32)?.to\_vec2::\<f32\>()? 一样。在加载模型时统一转为 DType::F32，即可在全链路任意节点直接通过 tensor.flatten\_all()?.to\_vec1::\<f32\>()?\[..8\] 无缝打印前若干个浮点数探针。

### **macOS 平台加载 Safetensors 与开启 Accelerate 的工程范式**

在编码实现上，有两种加载 .safetensors 并统一转换为 DType::F32 的标准方式。第一种方式是使用 candle\_core::safetensors::load 显式转换，最适合不依赖高层封装的纯白盒追踪器：由于 safetensors::load 会保持文件原始的 BF16 类型，可在加载循环中立即调用 .to\_dtype(DType::F32)? 完成转换。第二种方式是使用 VarBuilder::from\_mmaped\_safetensors 声明目标类型：只需在初始化 VarBuilder 时将第二个参数指定为 DType::F32，VarBuilder 会在每次调用 vb.get(...) 切片零拷贝映射文件时自动将张量转换为 DType::F32：  
`// 方式 A：直接加载 HashMap<String, Tensor> 并统一转换为 F32`  
`let raw_tensors = candle_core::safetensors::load(safetensors_path, &device)?;`  
`let mut weights = std::collections::HashMap::with_capacity(raw_tensors.len());`  
`for (name, tensor) in raw_tensors {`  
    `w[span_63](start_span)[span_63](end_span)eights.insert(name, tensor.to_dtype(candle_core::DType::F32)?);`  
`}`

`// 方式 B：通过内存映射 VarBuilder 自动在读取时转换为 F32`  
`let vb = unsafe {`  
    `candle_nn::VarBuilder::from_mmaped_safetensors(&[safetensors_path], candle_core::DType::F32, &device)?`  
`};`

此外，在 macOS（Apple Silicon）上配置 Cargo.\[span\_43\](start\_span)\[span\_43\](end\_span)toml 时，**强烈建议为 candle-core 与 candle-nn 开启 accelerate 特性**：  
`[dependencies]`  
`candle-core = { version = "0.11.0", features = ["accelerate"] }`  
`candle-nn = { version = "0.11.0", features = ["accelerate"] }`  
`safetensors = "0.8.0"`  
`tokenizers = "0.22.0"`

开启 features \= \["accelerate"\] 具有双重工程收益：一方面，它会调用 macOS 原生的 Apple Accelerate 框架（调度 Apple Silicon 的 AMX 矩阵协处理器），使 CPU 上的 F32 矩阵乘法吞吐量提升数倍；另一方面，如 candle Issue \#3744 实测所示，Accelerate 后端能够正确处理零步长广播批次的矩阵乘法，彻底规避基础 CPU 后端在边界步长布局下的潜在计算缺陷。

## **综合结论与工程对齐规范**

构建与 HuggingFace Python 逐位对齐（误差 \< 1e-5）的 Rust candle 白盒大语言模型前向计算追踪器，本质上是将 PyTorch 隐式代劳的步长重排、广播扩展与权重置换逻辑显式化。在线性投影阶段，由于 .safet\[span\_18\](start\_span)\[span\_18\](end\_span)ensors 权重 W 的存储形状为 \[d\_out, d\_in\] 且 candle-core 的 matmul 严格要求同秩输入，三维激活 x（\[B, T, d\_in\]）不可与二维 w.t()? 直接调用 .matmul()，而应采用 x.reshape((b \* t, d\_in))?.matmul(\&w.t()?)?.reshape((b, t, d\_out))? 的二维重塑范式，在避免报错的同时消除 broadcast\_matmul 带来的冗余权重拷贝。  
在注意力机制的张量流转中，多头拆分后的 .transpose(1, 2)? 会破坏物理内存连续性。此时，Q 和 K 在送入 RoPE 算子前必须显式调用 .contiguous()?，且在 HuggingFace .safetensors 格式下，SmolLM2、Llama-3.2 与 Qwen2.5 均统一对应 GPT-NeoX 非交错半区配对流派 (i, i \+ d\_h/2)，必须调用 candle\_nn::rotary\_emb::rope 并传入形状为 \[T, d\_h / 2\] 的 cos 与 sin 张量。对于不经过 RoPE 的 V 张量，则必须在 .transpose(1, 2)? 后（或通过 repeat\_kv 中的 unsqueeze(2)?.expand(...)?.reshape(...) 连续化扩展后）保持连续布局，以支持 att.matmul(\&v)；而 q.matmul(\&k.t()?) 中的 k.t()? 得益于底层 GEMM 对内层转置步长的原生支持，无需调用 .contiguous()?。配合加载模型时统一的 .to\_dtype(DType::F32) 转换与 macOS acce\[span\_44\](start\_span)\[span\_44\](end\_span)lerate 特性支持，白盒追踪器即可在保证执行效率的同时，实现全链路张量探针的高精度对齐。

#### **引用的文献**

1\. Chapter 5 — nn Modules & VarBuilder · candle Internals Handbook, https\://rumitx.com/books/candle/05-nn-modules-and-varbuilder.html 2\. Tensor in candle\_core \- Rust \- Docs.rs, https\://docs.rs/candle-core/latest/candle\_core/struct.Tensor.html 3\. Training a Language Model End-to-End in Rust: An Experience Report, https\://arxiv.org/pdf/2609.25008 4\. candle\_core/ tensor.rs \- Docs.rs, https\://docs.rs/candle-core/latest/src/candle\_core/tensor.rs.html 5\. quantized\_llama.rs uses wrong RoPE convention for NEOX-style, https\://github.com/huggingface/candle/issues/3410 6\. GitHub \- bmqube/minillm: A lightweight, efficient transformer, https\://github.com/bmqube/minillm 7\. CPU matmul silently returns incorrect values for stride-zero ... \- GitHub, https\://github.com/huggingface/candle/issues/3744 8\. 从零训练MNIST 手写数字识别：Candle 深度学习入门实战指南, https\://blog.csdn.net/gitblog\_01112/article/details/161800339 9\. candle\_transformers/models/ quantized\_llama.rs \- Docs.rs, https\://docs.rs/candle-transformers/latest/src/candle\_transformers/models/quantized\_llama.rs.html 10\. RustのcandleでMLを勉強する \- Zenn, https\://zenn.dev/mochineko/scraps/6215e74b830122 11\. candle\_transformers \- Rust \- Docs.rs, https\://docs.rs/candle-transformers 12\. Metal buffer pool grows beyond physical RAM on small unified, https\://github.com/huggingface/candle/issues/3464 13\. Module in candle\_nn \- Rust \- Docs.rs, https\://docs.rs/candle-nn/latest/candle\_nn/trait.Module.html