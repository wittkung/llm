# 前置篇｜拆解大模型前的两件趁手兵器：张量线性代数与 Rust 极简入门

很多人想彻底搞懂大语言模型（LLM）内部到底是怎么算的，但往往在第一步就被两座大山劝退了：

第一座山是**线性代数**。一翻开教材，满眼都是行列式、伴随矩阵、逆矩阵、特征值分解、若尔当标准型——还没看到神经网络，就已经被绕晕了。但真相是：**上面提到的这些概念，在大语言模型的前向推理（Inference）里，一个都用不到！**

第二座山是**编程语言**。为了不让框架黑盒遮住底层细节，我们这个系列用 **Rust**（配合 Hugging Face 极简张量库 `candle-core`）手写白盒探针。没写过 Rust 的朋友一听“所有权、生命周期、宏编程”就容易发怵。但真相同样很简单：**写一个工业级大模型推理探针，根本用不到复杂的生命周期标注和宏，只需要掌握 6 个最基础的语法结构，30 分钟就能看懂全部源码。**

这篇文章就是为你准备的“最小够用工具箱（Just-in-Time Toolkit）”：
- **上半场（第一至五节）**：只讲大模型推理真正用到的 **5 条张量与线性代数法则**。学会它们，第零篇里的每一处矩阵变形你都能用纸笔算出来。
- **下半场（第六至十节）**：专为有 Python / C / JS / Java 基础的读者准备的 **Rust 与 `candle` 30 分钟极简速成**。我们直接拿本系列第零篇的真实代码当例子，把你会遇到的每一个符号拆开讲透。

如果你已经熟悉矩阵维度变换与 Rust 基础语法，可以直接跳过本篇，前往**《第零篇｜一次大语言模型推理的完整旅程：从输入文本到下一个 Token》**。

本篇的所有数学与代码例子都内置在配套仓库（[https://github.com/wittkung/llm](https://github.com/wittkung/llm)）的 [`crates/llm-tracer/src/prelude.rs`](../../crates/llm-tracer/src/prelude.rs) 中。克隆仓库后，不需要下载任何模型文件，直接运行下面这行命令就能在终端看到本篇全部 5 组实验的真实输出：

```bash
git clone https://github.com/wittkung/llm.git
cd llm
cargo run -- --mode prelude
```

---

## 上半场：大模型推理只用到的 5 条张量与线性代数法则

---

## 一、 法则 1：从标量到四维张量——怎么一眼看懂 `[B, h, T, d_h]`？

在深度学习代码里，你每隔三行就会看到“张量（Tensor）”这个词，以及一串像 `[49152, 576]`、`[1, 10, 576]`、`[1, 9, 10, 64]` 这样的方括号数字。

别被“张量”这个数学名词唬住。在计算机程序里，**张量就是一个规定了形状（Shape / Dims）的多维浮点数（或整数）数组**。方括号里有几个数字，它就是几维（几阶）张量：

| 维度阶数 (Rank) | 数学名称 | 形状写法示例 | 在大模型里代表什么？ | 直观例子 |
|---|---|---|---|---|
| **0 维** | **标量 (Scalar)** | `[]` 或单个数字 | 单个数值（如缩放系数、最终挑出的单个词号） | `19556` 或 `0.125` |
| **1 维** | **向量 (Vector)** | `[d]`（如 `[4]` 或 `[576]`） | **一个词元（Token）的特征表示**，或者 RMSNorm 的一维缩放权重 | `[1.0, 0.0, 1.0, 0.0]`（用 4 个数字描述 1 个词） |
| **2 维** | **矩阵 (Matrix)** | `[行, 列]`（如 `[5, 4]` 或 `[49152, 576]`） | **模型的静态参数权重表**（如词嵌入表 `[词表大小 V, 特征宽度 d]`） | 一张有 5 行、每行 4 个数字的二维表格 |
| **3 维** | **三维张量** | **`[B, T, d]`**（如 `[1, 3, 4]` 或 `[1, 10, 576]`） | **流过模型主干的序列隐藏状态**（$B$ 句话，每句 $T$ 个词，每词 $d$ 个特征） | 把 $B=1$ 句话里的 $T=3$ 个词向量按顺序摞成一沓 |
| **4 维** | **四维张量** | **`[B, h, T, d_h]`**（如 `[1, 2, 3, 2]` 或 `[1, 9, 10, 64]`） | **多头注意力内部张量**（把每个词的 $d$ 维特征拆给 $h$ 个注意力头，每头分到 $d_h$ 维） | $B$ 句话、$h$ 个独立视角、每个视角看 $T$ 个词的 $d_h$ 维切片 |

我们拿 `prelude.rs` 里的第一组实验举例。假设我们的迷你词表里一共有 $V = 5$ 个词，每个词用 $d = 4$ 个浮点数表示，那么词嵌入表 `embed_table` 就是一个形状为 **`[5, 4]`** 的二维矩阵：

$$\mathbf{E}_{[5, 4]} = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 2.0 & 0.0 & 1.0 \\ 1.0 & 1.0 & 0.0 & 2.0 \\ 0.5 & 0.5 & 0.5 & 0.5 \end{bmatrix} \begin{matrix} \leftarrow \text{第 0 号词} \\ \leftarrow \text{第 1 号词} \\ \leftarrow \text{第 2 号词} \\ \leftarrow \text{第 3 号词} \\ \leftarrow \text{第 4 号词} \end{matrix}$$

现在用户输入了 $B = 1$ 句话，里面包含 $T = 3$ 个词，词号分别是 `[1, 2, 3]`（形状为 `[1, 3]`）。模型要做的第一件事（即第零篇的**阶段 02：词嵌入查表**），就是去上面那张 `[5, 4]` 的表里，把第 `1` 行、第 `2` 行、第 `3` 行原封不动地抽出，拼成一个形状为 **`[1, 3, 4]`**（即 `[B=1, T=3, d=4]`）的三维张量 $\mathbf{X}$：

```rust
// 对应 crates/llm-tracer/src/prelude.rs 第 [1/5] 节
let embed_table = Tensor::from_slice(
    &[
        0.1f32, 0.2, 0.3, 0.4, // 第 0 行
        1.0, 0.0, 1.0, 0.0,    // 第 1 行
        0.0, 2.0, 0.0, 1.0,    // 第 2 行
        1.0, 1.0, 0.0, 2.0,    // 第 3 行
        0.5, 0.5, 0.5, 0.5,    // 第 4 行
    ],
    (5, 4), // 指定形状为 [5, 4]
    dev,
)?;
let input_ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), dev)?;
let flat_ids = input_ids.flatten_all()?; // 展平为一维 [3]，供 index_select 查表
let x = embed_table.index_select(&flat_ids, 0)?.reshape((1, 3, 4))?;
```

运行 `cargo run -- --mode prelude`，你会看到终端打印出查表后的三维张量 `x`：

```text
--- [1/5] Tensor Dimensions & Embedding Row Lookup (index_select) ---
  embed_table shape: [5, 4] (Static Weight [V=5, d=4])
  input_ids   shape: [1, 3] (Token IDs [B=1, T=3])
  x (looked up)    : [1, 3, 4] -> [[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [1.0, 1.0, 0.0, 2.0]]
```

**读维度的小窍门：从左往右看，就是从外层括号往里层括号剥。**
看到 `x` 的形状是 `[1, 3, 4]`：
- 最外层有 `1` 个句子；
- 剥开一层，里面有 `3` 个词；
- 再剥开一层，每个词是一串长度为 `4` 的浮点数向量（如第 1 个词是 `[1.0, 0.0, 1.0, 0.0]`）。

---

## 二、 法则 2：向量点积（Dot Product）——大模型测量“两个词配不配”的唯一尺子

在大模型的注意力机制（Attention）里，当前词手握一个查询向量 $\mathbf{q}$（Query，代表“我想找什么信息”），历史上的每个词各举着一个键向量 $\mathbf{k}$（Key，代表“我这里有什么信息”）。

计算机怎么判断 $\mathbf{q}$ 和哪个 $\mathbf{k}$ 最匹配？答案是算它们的**向量点积（Dot Product，也叫内积）**。

### 1. 代数怎么算？对应位置相乘，再全部加起来

两个向量要做点积，**长度必须完全相等**。假设有两个长度为 2 的向量 $\mathbf{q} = [q_1, q_2]$ 和 $\mathbf{k} = [k_1, k_2]$，它们的点积就是一个标量数字：

$$\mathbf{q} \cdot \mathbf{k} = q_1 \times k_1 + q_2 \times k_2$$

### 2. 几何上为什么能测量相似度？

在几何上，两个向量的点积等于它们各自的长度（模长）乘以它们夹角 $\theta$ 的余弦值：

$$\mathbf{q} \cdot \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos\theta$$

因此：
- 当两个向量**方向完全相同**（夹角 $\theta = 0^\circ, \cos 0^\circ = 1$）时，点积达到**正的最大值**（高度匹配）；
- 当两个向量**互相垂直 / 正交**（夹角 $\theta = 90^\circ, \cos 90^\circ = 0$，毫不相干）时，点积等于 **`0`**；
- 当两个向量**方向完全相反**（夹角 $\theta = 180^\circ, \cos 180^\circ = -1$）时，点积是**负数**（强烈排斥）。

我们直接用 `prelude.rs` 里的第二组实验验证。设查询向量 $\mathbf{q} = [2.0, 0.0]$（指向正右方），去和三个候选键向量做点积：
- 候选 1：$\mathbf{k}_1 = [1.0, 1.0]$（指向右上 $45^\circ$）$\to$ 点积 $= 2.0 \times 1.0 + 0.0 \times 1.0 = \mathbf{2.0}$；
- 候选 2：$\mathbf{k}_2 = [2.0, 0.0]$（同样指向正右方，方向与 $\mathbf{q}$ 完全一致）$\to$ 点积 $= 2.0 \times 2.0 + 0.0 \times 0.0 = \mathbf{4.0}$（得分最高！）；
- 候选 3：$\mathbf{k}_3 = [0.0, 2.0]$（指向正上方，与 $\mathbf{q}$ 垂直 $90^\circ$）$\to$ 点积 $= 2.0 \times 0.0 + 0.0 \times 2.0 = \mathbf{0.0}$（毫不相关）。

```text
--- [2/5] Dot Product: Measuring Similarity (q · k) ---
  q [1, 2] · keys^T [2, 3] -> scores [1, 3] = [[2.0, 4.0, 0.0]]
```

记住这个直觉：整座大模型里最核心的注意力分数矩阵 $\mathbf{Q}\mathbf{K}^\top$，本质上不过是在批量计算每一对词向量之间的点积。

---

## 三、 法则 3：矩阵乘法（MatMul）与“中间维度抵消律”

整座大模型 99% 的计算时间都花在同一种运算上：**矩阵乘法（Matrix Multiplication，代码中写作 `matmul`）**。

要看懂所有线性投影层的形状变化，你只需要死死记住两条铁律：

### 1. 铁律一：中间维度必须相等并抵消，外侧维度保留

假设左矩阵 $\mathbf{A}$ 的形状是 **`[M, K]`**（$M$ 行 $K$ 列），右矩阵 $\mathbf{B}$ 的形状是 **`[K, N]`**（$K$ 行 $N$ 列）：

$$\underbrace{[M, \mathbf{K}]}_{\text{左矩阵 } \mathbf{A}} \times \underbrace{[\mathbf{K}, N]}_{\text{右矩阵 } \mathbf{B}} \longrightarrow \underbrace{[M, N]}_{\text{结果矩阵 } \mathbf{C}}$$

- **怎么算的？** 结果矩阵 $\mathbf{C}$ 的第 $i$ 行、第 $j$ 列的那个数字，就是拿左矩阵 $\mathbf{A}$ 的**第 $i$ 整行**（长度为 $K$ 的向量）与右矩阵 $\mathbf{B}$ 的**第 $j$ 整列**（长度也为 $K$ 的向量）做一次**向量点积**！
- **为什么中间的 $K$ 必须相等？** 因为左边的“行长”等于它的列数 $K$，右边的“列长”等于它的行数 $K$——如果这两个数不相等，点积就无法一一对应相乘。
- **怎么秒推输出形状？** 把两个矩阵的形状并排写在一起 `[M, K] × [K, N]`，像玩消消乐一样**把挨在一起的两个内侧维度 `K` 划掉**，剩下的外侧维度 **`[M, N]`** 就是输出矩阵的形状！

### 2. 铁律二：为什么权重矩阵在相乘前总是要调用转置 `.t()`？

在 PyTorch 和 Hugging Face 的 `.safetensors` 权重文件里，有一个统一的工业界约定：**线性层权重矩阵 $\mathbf{W}$ 的存储形状永远是 `[输出维度 d_out, 输入维度 d_in]`**。

比如在 `prelude.rs` 的第三组实验里，我们的输入 `x` 每个词有 `d_in = 4` 个特征，想通过线性层压成 `d_out = 2` 个特征。此时权重矩阵 `w_proj` 在文件里的形状是 **`[2, 4]`**：

$$\mathbf{W}_{\text{proj } [2, 4]} = \begin{bmatrix} 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 1.0 \end{bmatrix}$$

现在问题来了：如果我们的输入矩阵是 `[3, 4]`（3 个词，每词 4 维），能直接去乘 `[2, 4]` 的 `w_proj` 吗？
把形状并排摆出来看：`[3, 4] × [2, 4]` —— 中间挨着的两个数字一个是 `4`，一个是 `2`，**不相等，根本没法乘！**

怎么办？把权重矩阵 `w_proj` **转置（Transpose，记作 $\mathbf{W}^\top$，代码里写 `w_proj.t()?`）**！
什么叫矩阵转置？就是**把原来的第 $i$ 行竖过来变成第 $i$ 列**，行号和列号对调，形状从 `[2, 4]` 变成 **`[4, 2]`**：

$$\mathbf{W}_{\text{proj } [4, 2]}^\top = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$$

现在再把形状并排摆出来：

$$[3, \mathbf{4}] \times [\mathbf{4}, 2] \xrightarrow{\text{中间的 4 抵消}} [3, 2]$$

我们亲手算一下第 1 个词 `[1.0, 0.0, 1.0, 0.0]` 乘以 $\mathbf{W}_{\text{proj}}^\top$ 的两列：
- 乘第 1 列 `[1.0, 0.0, 1.0, 0.0]`：$1\times 1 + 0\times 0 + 1\times 1 + 0\times 0 = \mathbf{2.0}$；
- 乘第 2 列 `[0.0, 1.0, 0.0, 1.0]`：$1\times 0 + 0\times 1 + 1\times 0 + 0\times 1 = \mathbf{0.0}$。
所以第 1 个词投影后的结果就是 `[2.0, 0.0]`！

### 3. 三维激活 `[B, T, d_in]` 怎么和二维权重 `[d_in, d_out]` 相乘？

在实际推理中，输入 `x` 带着批大小 $B$，是三维张量 `[B, T, d_in]`（如 `[1, 3, 4]`）；而转置后的权重 `weight.t()?` 是二维矩阵 `[4, 2]`。在 `candle-core` 中，`matmul` 要求左右两边的维度阶数（Rank）相同。

因此，我们在 [`src/ops.rs`](../../crates/llm-tracer/src/ops.rs) 的 `linear_no_bias` 函数里用了最经典的三步法：
1. 先用 `.reshape((b * t, d_in))?` 把前两维（句子数 $B$ 和词数 $T$）合并成二维矩阵 **`[B * T, d_in]`**（即总共 $B \times T$ 个词，排成一张大表）；
2. 做标准二维矩阵乘法：`[B * T, d_in] × [d_in, d_out] -> [B * T, d_out]`；
3. 再用 `.reshape((b, t, d_out))?` 还原回三维张量 **`[B, T, d_out]`**：

```rust
// crates/llm-tracer/src/ops.rs 中的算子 1
pub fn linear_no_bias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (b, t, d_in) = x.dims3()?;
    let (d_out, _) = weight.dims2()?;
    x.reshape((b * t, d_in))?
        .matmul(&weight.t()?)?
        .reshape((b, t, d_out))
}
```

---

## 四、 法则 4：千万别混的两种乘法——矩阵乘法 `matmul` vs. 逐元素乘法 `*`

读大模型公式和代码时，新手最容易搞混的莫过于这两种乘法：

| 对比维度 | **矩阵乘法（Matrix Multiplication）** | **逐元素乘法（Element-wise / Hadamard Product）** |
|---|---|---|
| **数学符号** | $\mathbf{A}\mathbf{B}$ 或 $\mathbf{A} \times \mathbf{B}$（中间不写点或写叉号） | $\mathbf{A} \odot \mathbf{B}$（带圈的点） |
| **Rust `candle` 写法** | `a.matmul(&b)?` | `(&a * &b)?`（直接用星号 `*`） |
| **输入形状要求** | 内侧维度相等：`[M, K]` 与 `[K, N]` | **两个张量的形状必须一模一样**：`[M, N]` 与 `[M, N]` |
| **输出形状与算法** | 输出 `[M, N]`；行与列做点积求和 | 输出仍是 `[M, N]`；**相同坐标位置上的两个数直接相乘，互不串门** |
| **在第零篇哪里用到？** | QK 点积得分、线性层投影（阶段 04/05/06/07/09） | **阶段 07 SwiGLU MLP 的门控相乘**：`(&gate * &up)?` |

看 `prelude.rs` 里的第四组实验：同样拿两个 $2 \times 2$ 的矩阵 `gate` 和 `up`，分别做逐元素乘法 `(&gate * &up)?` 和矩阵乘法 `gate.matmul(&up)?`：

$$\text{gate} = \begin{bmatrix} 0.5 & 2.0 \\ 0.0 & 1.5 \end{bmatrix}, \quad \text{up} = \begin{bmatrix} 4.0 & 3.0 \\ 9.0 & 2.0 \end{bmatrix}$$

- **逐元素乘法 `(&gate * &up)?`（对应位置直接乘）**：
  $$\begin{bmatrix} 0.5 \times 4.0 & 2.0 \times 3.0 \\ 0.0 \times 9.0 & 1.5 \times 2.0 \end{bmatrix} = \begin{bmatrix} \mathbf{2.0} & \mathbf{6.0} \\ \mathbf{0.0} & \mathbf{3.0} \end{bmatrix}$$
  你看左下角：因为 `gate` 左下角是 `0.0`（相当于阀门关死），不管 `up` 左下角是 `9.0` 还是 `900.0`，乘出来立刻变成 `0.0`——这就是第零篇阶段 07“门控（Gating）”名字的由来！
- **矩阵乘法 `gate.matmul(&up)?`（行乘列再相加）**：
  $$\begin{bmatrix} 0.5\times 4.0 + 2.0\times 9.0 & 0.5\times 3.0 + 2.0\times 2.0 \\ 0.0\times 4.0 + 1.5\times 9.0 & 0.0\times 3.0 + 1.5\times 2.0 \end{bmatrix} = \begin{bmatrix} \mathbf{20.0} & \mathbf{5.5} \\ \mathbf{13.5} & \mathbf{3.0} \end{bmatrix}$$

---

## 五、 法则 5：形状重排、内存步长（Stride）、`.contiguous()` 与广播

在多头注意力（阶段 04 和 05）的代码里，你会看到一行非常经典的链式调用：

```rust
let q = q_lin.reshape((b, seq_len, h_q, d_h))?.transpose(1, 2)?.contiguous()?;
```

很多写过好几年 PyTorch 的人也说不清：**`reshape` 和 `transpose` 到底有什么区别？为什么 `transpose(1, 2)` 后面非要加一个 `.contiguous()`？**

要彻底搞懂这件事，你只需要知道计算机内存的一个物理真相：**计算机内存条是一维的，根本没有二维或四维的物理格子。**

无论一个张量的逻辑形状是 `[3, 4]`、`[1, 3, 2, 2]` 还是 `[1, 2, 3, 2]`，它在物理内存里永远是一字排开的 **12 个连续浮点数**：

```text
物理内存一维数组（共 12 个数）：
下标:  0    1    2    3    4    5    6    7    8    9   10   11
数值: [1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0]
       └── 词0,头0 ──┘ └── 词0,头1 ──┘ └── 词1,头0 ──┘ └── 词1,头1 ──┘ ...
```

那张量怎么知道自己是几维的？靠它随身携带的两个小账本：**形状（`dims`）** 和 **步长（`stride`，即沿着某个维度下标加 1 时，在物理一维数组里需要跨过几个元素）**。

我们结合 `prelude.rs` 第五组实验的真实输出，看清这三步到底干了什么：

```text
--- [5/5] Split Heads (reshape + transpose + contiguous) & GQA Broadcast ---
  After reshape [1, 3, 2, 2]: is_contiguous = true | strides = [12, 4, 2, 1]
  After transpose(1, 2) [1, 2, 3, 2]: is_contiguous = false | strides = [12, 2, 4, 1]
  After .contiguous()   [1, 2, 3, 2]: is_contiguous = true | strides = [12, 6, 2, 1]
```

1. **第一步 `.reshape((1, 3, 2, 2))`（零拷贝重塑视图）**：
   把每个词长度为 `4` 的向量，原地切成 `h=2` 个头、每头 `d_h=2` 个数，形状从 `[1, 3, 4]` 变成 **`[1, 3, 2, 2]`**（含义：`[1句话, 3个词, 2个头, 每头2维]`）。
   注意看此时物理内存动了吗？**完全没动！** 步长 `strides = [12, 4, 2, 1]` 从左到右递减，说明按逻辑顺序遍历元素时，物理内存下标正好是 `0, 1, 2, 3...` 顺着往下走，这种状态叫**内存连续（`is_contiguous = true`）**。
2. **第二步 `.transpose(1, 2)`（交换第 1 维和第 2 维）**：
   为了让不同注意力头各自独立做矩阵乘法，我们必须把“头数 `h=2`”提到“词数 `T=3`”前面，让形状从 `[1, 3, 2, 2]` 变成 **`[1, 2, 3, 2]`**（`[1句话, 2个头, 3个词, 每头2维]`）。
   注意：调用 `.transpose(1, 2)` 时，为了速度，它**并没有去搬移物理内存里的数字**，而是直接把账本上的第 1 维和第 2 维步长对调，变成了 `strides = [12, 2, 4, 1]`！
   这会导致什么后果？现在你想顺着读“第 0 号头的全部 3 个词”，物理内存却得跳着读：先读下标 `0, 1`（词 0 头 0），然后**跳过词 0 头 1** 去读下标 `4, 5`（词 1 头 0），再跳去读下标 `8, 9`（词 2 头 0）！物理内存不挨着了，所以 **`is_contiguous` 变成了 `false`**。
3. **第三步 `.contiguous()`（按新形状把物理内存真正排整齐）**：
   后续的 RoPE 旋转算子要求每个头的数据在物理内存里必须紧挨着存放。调用 `.contiguous()` 后，程序才真正分配一块新内存，把“头 0 的 3 个词”排在前面（下标 `0..6`），“头 1 的 3 个词”排在后面（下标 `6..12`），步长恢复为从大到小递减的 `[12, 6, 2, 1]`，`is_contiguous` 重新变回 `true`。

### 附：什么是广播（Broadcasting，`unsqueeze` + `expand`）？

在第零篇阶段 05 的分组查询注意力（GQA）里，我们有 9 个查询头，但只有 3 个键值头。为了让每 1 个键值头复制 3 份去配对 3 个查询头，我们在 `repeat_kv` 里写了三行代码：

```rust
// 假设输入 x 形状为 [B, h_kv, T, d_h]，例如 [1, 1, 3, 2]
x.unsqueeze(2)?                            // 1. 在第 2 维插入一个长度为 1 的新轴 -> [1, 1, 1, 3, 2]
    .expand((b, h_kv, n_rep, seq_len, d_h))? // 2. 将这个长度为 1 的轴零拷贝广播扩展为 n_rep=3 -> [1, 1, 3, 3, 2]
    .reshape((b, h_kv * n_rep, seq_len, d_h)) // 3. 合并第 1 维和第 2 维 -> [1, 3, 3, 2]
```

这就是张量的**广播机制**：当某个维度的长度是 `1` 时，可以通过 `.expand(...)` 把它沿着该维度重复展开成任意长度 $N$，从而与另一个维度为 $N$ 的张量对齐运算。

---

## 下半场：给 Python / C / JS 开发者的 Rust 与 `candle` 30 分钟速成

看完了上面的 5 条数学法则，你可能已经发现：Rust 写出来的张量代码和 Python（PyTorch / NumPy）长得有八成相似。下面我们把剩下的两成语法差异——也是新手读 Rust 源码最容易产生疑问的 **5 个核心语法点**——一次性扫清。

---

## 六、 语法 1：变量绑定 `let`、不可变默认规则与类型推导

在 [`src/model.rs`](../../crates/llm-tracer/src/model.rs) 里，你会看到两种声明变量的方式：

```rust
let d = cfg.hidden_size; // 不可变绑定：后面永远不能修改 d 的值
let mut hidden = w_embed.index_select(&flat_ids, 0)?.reshape((b, seq_len, d))?; // 可变绑定 (mut)
```

1. **默认只读（Immutable by Default）**：
   在 Rust 里，用 `let x = ...` 定义的变量**默认是只读常量**！如果后续代码不小心改了 `d`，编译器会直接报错。这在阅读长代码时极其省心：只要你看到一个变量声明时没有写 `mut`，你就百分之百确定它在整段函数里绝不会被任何地方偷偷篡改。
2. **需要更新时显式加上 `mut`（Mutable）**：
   像残差流主干张量 `hidden`，每穿过一层 Transformer 都要更新一次（`hidden = (&h_mid + &mlp_out)?;`），所以声明时必须显式写上 `let mut hidden = ...`。
3. **自动类型推导与元组解构（Tuple Destructuring）**：
   虽然 Rust 是强类型静态编译语言，但你几乎不需要手写繁琐的类型名，编译器会自动推导。同时它支持像 Python 一样一行解构元组：
   ```rust
   let (b, seq_len) = input_ids.dims2()?; // 将二维张量的形状直接解构为批大小 b 和序列长度 seq_len
   ```

---

## 七、 语法 2：所有权与借用——为什么张量相乘要写 `&gate * &up`？

如果你以前写 Python 或 C++，看到下面这行代码一定会好奇——为什么两个张量相乘或相加，前面都要加一个取地址符（引用符）`&`？

```rust
let gated = (&gate * &up)?;
let h_mid = (&hidden + &attn_out)?;
```

这背后就是 Rust 最著名的**所有权（Ownership）与借用（Borrowing）**机制。完全不用背概念，用一句话就能讲明白：

- **如果直接传值 `func(x)`（不加 `&`）：交出所有权（Move）**
  在 Rust 中，像 `Tensor`、`String`、`Vec` 这样管理着堆内存的对象，如果直接把变量名传给一个函数（或者直接写 `gate * up`），相当于**把变量的所有权移交给了对方**，用完就销毁了，下一行代码你就再也不能访问 `gate` 或 `hidden` 了！
- **如果传不可变引用 `&x`（加上 `&`）：只读借用（Shared Borrow）**
  在变量前加个 `&`（如 `&gate`、`&hidden`、`linear_no_bias(&x_norm, w_q)`），意思是：**“我只把数据的只读视图借给你看一眼/算一下，既不要在内存里复制整张大矩阵，算完之后也别销毁我的原张量，我后面还要接着用它呢！”**（例如 `hidden` 在算完 `x_norm` 后，还要留到阶段 06 做残差相加 `&hidden + &attn_out`）。
- **如果传可变引用 `&mut x`：可写借用（Mutable Borrow）**
  当一个函数不仅要读某个变量，还要往里面写入新数据时，就传 `&mut`。比如第零篇里 30 层的 KV Cache 槽位，每跑完一步都要把新算出的 $\mathbf{K}, \mathbf{V}$ 存进去，所以函数签名和调用处都显式标出了 `&mut kv_caches`——一眼就能看出哪个参数会被原地修改。

---

## 八、 语法 3：结构体 `struct`、枚举 `enum` 与模式匹配 `match`

Rust 没有传统面向对象语言里复杂的类继承（`class A extends B`），它只用两个干净的数据结构来组织一切：**结构体（`struct`，把几个字段打包在一起）** 和 **枚举（`enum`，几个互斥选项里选一个）**。

### 1. 结构体 `struct` 与方法块 `impl`

打开 [`src/inspect.rs`](../../crates/llm-tracer/src/inspect.rs)，我们用 `struct LlamaConfig` 映射 `config.json` 里的超参数，再用 `impl LlamaConfig` 给它挂载成员方法：

```rust
#[derive(Debug, Clone, Deserialize)] // 自动生成打印(Debug)、拷贝(Clone)和 JSON 反序列化代码
pub struct LlamaConfig {
    pub hidden_size: usize,         // d = 576
    pub num_attention_heads: usize, // h_q = 9
    // ...
}

impl LlamaConfig {
    // &self 相当于 Python 里的 self 或 C++/Java 里的 this
    // 注意函数最后一行不加分号 (;)，代表直接把这行算出的值作为函数返回值！
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
```

这里顺带记住 Rust 的一个贴心简写：**函数里最后一行表达式如果不写分号 `;`，它就是这个函数的返回值**（等同于写了 `return self.hidden_size / self.num_attention_heads;`）。

### 2. 枚举 `enum`、模式匹配 `match` 与 `Option<T>`

打开 [`src/ops.rs`](../../crates/llm-tracer/src/ops.rs)，我们定义了张量的三种角色 `Role`，并用 `match`（加强版的 `switch-case`，编译器会强制检查你有没有漏掉任何一个分支）把它转成打印标签：

```rust
pub enum Role {
    Weight,     // 静态权重
    Activation, // 动态激活
    KvCache,    // 键值缓存
}

let role_tag = match role {
    Role::Weight     => "WEIGHT    ",
    Role::Activation => "ACTIVATION",
    Role::KvCache    => "KV_CACHE  ",
};
```

更重要的是，**Rust 语言里没有 `null` / `None` 空指针异常！**
当一个槽位“可能装了数据，也可能还是空的”——比如第 1 轮 Prefill 开始前，每一层的 KV Cache 还是空的；第 2 轮 Decode 时，里面已经有上一轮存好的 `(prev_k, prev_v)` 了——Rust 使用标准库内置的枚举 **`Option<T>`**，它只有两个分支：
- `Some(数据)`：槽位里有东西；
- `None`：槽位目前是空的。

回头看 [`src/model.rs`](../../crates/llm-tracer/src/model.rs) 里更新 KV Cache 的那 5 行代码，现在是不是像读白话文一样清楚？

```rust
// kv_slot.take() 把槽位里的 Option<(Tensor, Tensor)> 取出来：
let (k_cache, v_cache) = match kv_slot.take() {
    // 如果里面已经有历史缓存 Some((prev_k, prev_v))，就把新算出的 k_rope/v 拼在历史缓存后面（第 2 维）
    Some((prev_k, prev_v)) => (
        Tensor::cat(&[&prev_k, &k_rope], 2)?,
        Tensor::cat(&[&prev_v, &v], 2)?,
    ),
    // 如果槽位是空的 (None)，说明现在是第 1 轮 Prefill，直接拿当前的 k_rope 和 v 当作初始缓存
    None => (k_rope, v),
};
```

---

## 九、 语法 4：错误处理 `Result<T>` 与行尾那个神出鬼没的问号 `?`

初看本系列代码时，你一定会问：**为什么几乎每一行张量计算的末尾，都在小括号后面跟着一个问号 `?`？**

```rust
let q_lin = linear_no_bias(&x_norm, w_q)?;
let q = q_lin.reshape((b, seq_len, h_q, d_h))?.transpose(1, 2)?.contiguous()?;
```

在张量计算中，随时可能因为形状不匹配、文件不存在或维度越界而出错。Rust 没有采用 Python/Java 那种容易在半路突然崩溃跳出的 `try / catch` 异常机制，而是让所有可能出错的函数都返回一个枚举类型 **`Result<成功值类型>`**，它也只有两个分支：
- `Ok(运算结果)`：计算成功，里面包着算出来的 `Tensor`；
- `Err(错误原因)`：计算出错，里面包着详细的报错信息（比如哪两个维度没对齐）。

如果每次调函数都要手写 `match` 去拆开 `Ok` 和 `Err`，代码就会变得又臭又长。于是 Rust 发明了极其优雅的**问号运算符 `?`**：
- 当你在返回 `Result` 的函数调用后面加一个 **`?`** 时：
  - 如果结果是 **`Ok(val)`**，`?` 会自动剥掉外壳，把里面的 `val`（比如 `Tensor`）取出来赋给左边的变量，继续往下走；
  - 如果结果是 **`Err(e)`**，`?` 会立刻中止当前函数，把错误 `Err(e)` 原封不动地向上抛给调用者！

也就是说，行尾那一个小小的 **`?`**，等于一行代码帮你写完了“如果成功就拆包取值，如果失败就立刻带着错误退出函数”的全部安全检查。

---

## 十、 语法 5：迭代器链 `.map().collect()` 与 PyTorch / `candle` 算子对照表

### 1. 一行代码生成数组：`(0..half).map(...).collect()`

在 Python 里，我们经常写列表推导式（List Comprehension），比如：
```python
# Python 列表推导式：生成 [0.0, 1.0, 2.0, ..., max_seq_len - 1]
positions = [float(p) for p in range(max_seq_len)]
```

在 Rust 中，完全等价的写法是**范围（`0..n`）+ 映射闭包（`.map(|参数| 表达式)`）+ 收集为数组（`.collect()`）**。看 [`src/ops.rs`](../../crates/llm-tracer/src/ops.rs) 里生成 RoPE 位置序列和逆频率表的代码：

```rust
// 0..max_seq_len 相当于 Python 的 range(max_seq_len)
// |p| p as f32 是匿名函数（闭包），相当于 Python 的 lambda p: float(p)
// .collect() 把流式算出的每个元素打包进 Vec<f32> 动态数组里
let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
```

### 2. PyTorch / NumPy vs. Rust `candle-core` 常用张量 API 一览表

最后，我们把第零篇全部 600 行源码中调用过的 `candle` 张量方法整理成一张速查表。只要你用过 NumPy 或 PyTorch，看一眼就能直接对应上：

| 操作目的 | Python (PyTorch) 写法 | Rust (`candle-core` / `candle-nn`) 写法 | 在第零篇哪里用到？ |
|---|---|---|---|
| **从数组创建张量** | `torch.tensor(slice).reshape(5, 4)` | `Tensor::from_slice(&slice, (5, 4), dev)?` | `prelude.rs`、`toy.rs`、`model.rs` |
| **查看张量形状** | `x.shape` | `x.dims()`（或拆解三维 `x.dims3()?`） | 全文各阶段维度校验 |
| **转换浮点精度** | `v.to(torch.float32)` | `v.to_dtype(DType::F32)?` | `inspect.rs` 将 BF16 权重转为 F32 |
| **按行号查表** | `w_embed[flat_ids]` 或 `F.embedding` | `w_embed.index_select(&flat_ids, 0)?` | **阶段 02**：Token ID 查词嵌入表 |
| **改变形状 (零拷贝)** | `x.reshape(b, seq_len, h, d_h)` | `x.reshape((b, seq_len, h, d_h))?` | **阶段 04/06**：拆头与合并多头 |
| **交换最后两维 (转置)** | `w.T` 或 `k.transpose(-2, -1)` | `w.t()?` 或 `k_full.t()?` | **阶段 04/05**：权重转置与 $\mathbf{Q}\mathbf{K}^\top$ |
| **交换指定两维** | `q.transpose(1, 2)` | `q.transpose(1, 2)?` | **阶段 04/06**：将头数维度 `h` 提前或还原 |
| **整理物理内存连续性** | `q.contiguous()` | `q.contiguous()?` | **阶段 04/06**：`transpose(1, 2)` 之后调用 |
| **矩阵乘法 (MatMul)** | `a @ b` 或 `torch.matmul(a, b)` | `a.matmul(&b)?` | 所有线性层投影、注意力得分与加权求和 |
| **逐元素运算** | `gate * up`、`hidden + attn_out` | `(&gate * &up)?`、`(&hidden + &attn_out)?` | **阶段 06/07**：残差相加与 SwiGLU 门控 |
| **沿指定维度切片** | `cos_all[offset : offset + seq_len]` | `cos_all.narrow(0, offset, seq_len)?` | **阶段 03**：切出当前位置的 RoPE 旋转表 |
| **沿指定维度拼接** | `torch.cat([prev_k, k_rope], dim=2)` | `Tensor::cat(&[&prev_k, &k_rope], 2)?` | **阶段 05 / 第十节**：追加拼接 KV Cache |
| **插入维度与广播展开** | `x.unsqueeze(2).expand(...)` | `x.unsqueeze(2)?.expand(...)` | **阶段 05**：GQA `repeat_kv` 键值头广播 |
| **取单个下标切片** | `logits[0, seq_len - 1]` | `logits.i((0, seq_len - 1))?` | **阶段 10**：取出最后一个词的 49152 维得分 |
| **求最大值下标** | `last_logits.argmax(dim=0).item()` | `last_logits.argmax(0)?.to_scalar::<u32>()?` | **阶段 10**：贪心选出得分最高的下一个词号 |

---

## 十一、 兵器已备齐，准备进入大模型内部！

恭喜你！到这里，你已经拿齐了拆解大模型所需的全部数学与代码武器：
- 在数学上，你知道了张量维度怎么一层层剥开，知道了点积为什么能测相似度，记住了矩阵乘法 `[M, K] × [K, N] -> [M, N]` 的消维铁律和 `.t()` 转置约定，分清了 `matmul` 与逐元素 `*`，也看懂了 `reshape`、`transpose(1, 2)` 和 `.contiguous()` 在一维物理内存上的真实动作；
- 在 Rust 语法上，你认识了 `let / let mut`、`&` 只读借用与 `&mut` 可变借用、`struct / enum / match`、处理空槽位的 `Option`、处理报错的 `Result` 与 `?` 运算符，以及 `candle-core` 的常用算子。

现在，打开 **[《第零篇｜一次大语言模型推理的完整旅程：从输入文本到下一个 Token》](../ep00-inference-journey/draft.md)**，我们正式把真实的大语言模型装上解剖台！
