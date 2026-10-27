# 前置篇｜线性代数与 Rust 极简入门

在开始后续实验之前，我们用这一篇独立的先导文章，把接下来会反复用到的**线性代数知识**与 **Rust 编程基础**一次性讲透。

大学线性代数课程通常花费大量学时讲解行列式、伴随矩阵、逆矩阵与特征值分解；好消息是，在大模型前向推理计算中，上述概念全部用不到。同样，用 Rust 编写张量计算代码与 Python（NumPy / PyTorch）高度相似，只需掌握最核心的 **6 个基础语法点**即可无障碍阅读后续全部源码。

- **上半场**（第一至六节）：多维张量计算的 **6 条线性代数与数值法则**；
- **下半场**（第七至十二节）：面向有其他语言基础读者的 **Rust 与 `candle` 张量库极简入门**。

本篇的所有数学与代码实验均收录在配套仓库（[https://github.com/wittkung/llm](https://github.com/wittkung/llm)）的 [`crates/llm-tracer/src/prelude.rs`](../../crates/llm-tracer/src/prelude.rs) 中。克隆仓库后运行以下命令即可查看全部 6 组实验输出：

```bash
git clone https://github.com/wittkung/llm.git
cd llm
cargo run -- --mode prelude
```

---

## 一、 标量、向量、矩阵与多维张量

在数值计算代码中，带形状（Shape）的多维数组统称为**张量**（Tensor）。方括号里包含几个维度数字，它就是几阶（Rank）张量：

| 阶数 (Rank) | 数学名称 | 形状示例 | 元素总数 |
|---|---|---|---|
| **0 阶** | 标量 (Scalar) | `[]` | $1$ 个 |
| **1 阶** | 向量 (Vector) | `[4]` | $4$ 个 |
| **2 阶** | 矩阵 (Matrix) | `[5, 4]` | $5 \times 4 = 20$ 个 |
| **3 阶** | 三阶张量 | `[1, 3, 4]` | $1 \times 3 \times 4 = 12$ 个 |
| **4 阶** | 四阶张量 | `[1, 2, 3, 2]` | $1 \times 2 \times 3 \times 2 = 12$ 个 |

各阶张量在几何结构上的对应关系如下：

- **0 阶标量 `[]`**：单独的一个浮点数（如 `3.14`），无方括号包裹；
- **1 阶向量 `[4]`**：一排连续排列的 4 个数字（如 `[1.0, 0.0, 1.0, 0.0]`）；
- **2 阶矩阵 `[5, 4]`**：一张 5 行 4 列的二维表格（行号 `0..4`，列号 `0..3`）；
- **3 阶张量 `[1, 3, 4]`**：1 组 3 行 4 列的二维表格（常见于 `[Batch, SeqLen, Dim]`）；
- **4 阶张量 `[1, 2, 3, 2]`**：外层 1 个大组，内含 2 个小组，每组是一张 $3 \times 2$ 表格（常见于多头注意力 `[Batch, Heads, SeqLen, HeadDim]`）。

> **维度阅读法则**：从左往右对应从最外层括号向最内层括号逐层展开；所有维度数字相乘即为张量包含的元素总数。

例如，给定一张 5 行 4 列的二维矩阵 `table`（形状 `[5, 4]`）：

$$
\mathbf{T}_{[5, 4]} = \begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 \\
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 2.0 & 0.0 & 1.0 \\
1.0 & 1.0 & 0.0 & 2.0 \\
0.5 & 0.5 & 0.5 & 0.5
\end{bmatrix}
\begin{matrix}
\leftarrow \text{第 0 行} \\
\leftarrow \text{第 1 行} \\
\leftarrow \text{第 2 行} \\
\leftarrow \text{第 3 行} \\
\leftarrow \text{第 4 行}
\end{matrix}
$$

若要按行号索引 `row_indices = [1, 2, 3]` 从第 0 维（行维度）抽出第 `1, 2, 3` 行拼成 `[3, 4]` 子表，再在外层套一层括号变为三维张量 `[1, 3, 4]`，只需调用 `index_select` 与 `reshape`：

```rust
// 对应 crates/llm-tracer/src/prelude.rs 第 [1/6] 节
let table = Tensor::from_slice(
    &[
        0.1f32, 0.2, 0.3, 0.4, // 第 0 行
        1.0, 0.0, 1.0, 0.0,    // 第 1 行
        0.0, 2.0, 0.0, 1.0,    // 第 2 行
        1.0, 1.0, 0.0, 2.0,    // 第 3 行
        0.5, 0.5, 0.5, 0.5,    // 第 4 行
    ],
    (5, 4),
    dev,
)?;
let row_indices = Tensor::from_slice(&[1u32, 2, 3], (3,), dev)?;
let picked_2d = table.index_select(&row_indices, 0)?; // [3, 4]
let tensor_3d = picked_2d.reshape((1, 3, 4))?;        // [1, 3, 4]
```

```text
--- [1/6] Tensor Shapes (1D / 2D / 3D) & Row Lookup (index_select) ---
  2D table     shape: [5, 4]
  row_indices  shape: [3]
  tensor_3d    shape: [1, 3, 4] -> [[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [1.0, 1.0, 0.0, 2.0]]
```

---

## 二、 向量点积与高维近似正交

给定两个长度相同的一维向量 $\mathbf{a} = [a_1, \dots, a_k]$ 与 $\mathbf{b} = [b_1, \dots, b_k]$，它们的**向量点积**（Dot Product / 内积）定义为相同下标元素两两相乘再求和：

$$
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_k b_k = \sum_{i=1}^{k} a_i b_i
$$

只有长度相等的两个向量才能做点积，且运算结果是一个**标量数字**。

### 1. 几何含义：方向相似度

在几何空间中，两个向量的点积等于它们各自的模长乘以夹角 $\theta$ 的余弦值：

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta
$$

因此，点积天然用于度量两个向量的方向相似度：

- **方向一致**（$\theta = 0^\circ$）：$\cos 0^\circ = 1$，点积取得正最大值；
- **夹角为锐角**（$0^\circ < \theta < 90^\circ$）：点积为正，方向越接近数值越大；
- **互相正交**（$\theta = 90^\circ$）：$\cos 90^\circ = 0$，点积恒为 **`0`**（两方向完全独立、互不相关）；
- **夹角为钝角或反向**（$90^\circ < \theta \le 180^\circ$）：点积为负。

以 `prelude.rs` 第二组实验为例，设 $\mathbf{a} = [2.0, 0.0]$（指向正右方），分别与三个向量做点积：

- $\mathbf{b}_1 = [1.0, 1.0]$（右上 $45^\circ$）：$\mathbf{a} \cdot \mathbf{b}_1 = 2.0 \times 1.0 + 0.0 \times 1.0 = \mathbf{2.0}$；
- $\mathbf{b}_2 = [2.0, 0.0]$（同向 $0^\circ$）：$\mathbf{a} \cdot \mathbf{b}_2 = 2.0 \times 2.0 + 0.0 \times 0.0 = \mathbf{4.0}$；
- $\mathbf{b}_3 = [0.0, 2.0]$（垂直 $90^\circ$）：$\mathbf{a} \cdot \mathbf{b}_3 = 2.0 \times 0.0 + 0.0 \times 2.0 = \mathbf{0.0}$。

```text
--- [2/6] Vector Dot Product (a · b) ---
  a [1, 2] · b_list^T [2, 3] -> [1, 3] = [[2.0, 4.0, 0.0]]
```

### 2. 高维近似正交与 Johnson-Lindenstrauss 引理

当两个单位向量满足**严格正交**（Orthogonal，$\mathbf{u} \cdot \mathbf{v} = 0$）时，沿这两个方向存储的信息在点积检索时完全不会互相干扰。在 $d$ 维欧氏空间 $\mathbb{R}^d$ 中，两两严格正交的单位向量最多只有 **$d$ 根**（即一组标准正交基）。

但如果允许一个极小的点积误差 $\epsilon$，只要求两两满足 $|\mathbf{u}_i \cdot \mathbf{u}_j| \le \epsilon$（夹角落在 $90^\circ$ 附近极窄区间内），则称为**近似正交**（Almost Orthogonal）。

高维几何中的 **Johnson-Lindenstrauss 引理**证明：在 $d$ 维空间中，两两满足 $|\mathbf{u}_i \cdot \mathbf{u}_j| \le \epsilon$ 的单位向量最大数量 $M$ 随维度 $d$ 呈**指数级增长**：

$$
M \sim \exp(c \cdot \epsilon^2 d)
$$

当维度 $d$ 达到数百或数千（如 $d = 576$）时，空间虽仅有 $576$ 根严格正交坐标轴，却能同时容纳**成千上万根近似正交的特征方向**。这正是高维向量空间能够将远多于维度数 $d$ 的稀疏特征压缩叠加在一起的数学基础。

---

## 三、 矩阵乘法、转置与正交旋转

**矩阵乘法**（Matrix Multiplication，代码中为 `matmul`） 是将左矩阵的每一行与右矩阵的每一列两两配对，批量计算向量点积。

### 1. 内侧维度消去律

设左矩阵 $\mathbf{A}$ 形状为 `[M, K]`，右矩阵 $\mathbf{B}$ 形状为 `[K, N]`：

$$
\underbrace{[M, \mathbf{K}]}_{\text{左矩阵 } \mathbf{A}} \times \underbrace{[\mathbf{K}, N]}_{\text{右矩阵 } \mathbf{B}} \longrightarrow \underbrace{[M, N]}_{\text{结果矩阵 } \mathbf{C}}, \qquad C_{i, j} = \sum_{k=1}^{K} A_{i, k} B_{k, j}
$$

- **合法前提**：左矩阵列数与右矩阵行数必须相等（同为 `K`），否则行向量与列向量长度不匹配，无法计算点积。
- **形状推导**：将 `[M, K] × [K, N]` 并排书写，划去紧挨着的内侧维度 `K`，保留外侧维度即得输出形状 **`[M, N]`**。

### 2. 矩阵转置（`.t()`）

**矩阵转置**（记作 $\mathbf{W}^\top$，代码中为 `w.t()?`） 是将矩阵的第 $i$ 行变为第 $i$ 列，形状从 `[M, N]` 反转为 **`[N, M]`**。

在 `prelude.rs` 第三组实验中，输入矩阵 $\mathbf{P}$ 形状为 `[3, 4]`，权重矩阵 $\mathbf{W}$ 形状为 `[2, 4]`：

$$
\mathbf{P}_{[3, 4]} = \begin{bmatrix}
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 2.0 & 0.0 & 1.0 \\
1.0 & 1.0 & 0.0 & 2.0
\end{bmatrix}, \quad
\mathbf{W}_{[2, 4]} = \begin{bmatrix}
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 1.0
\end{bmatrix}
$$

`[3, 4]` 与 `[2, 4]` 内侧维度不匹配；先将 $\mathbf{W}$ 转置为 $\mathbf{W}^\top: [4, 2]$，即可执行 `[3, 4] × [4, 2] -> [3, 2]`：

$$
\mathbf{P}_{[3, 4]} \times \mathbf{W}_{[4, 2]}^\top =
\begin{bmatrix}
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 2.0 & 0.0 & 1.0 \\
1.0 & 1.0 & 0.0 & 2.0
\end{bmatrix}
\begin{bmatrix}
1.0 & 0.0 \\
0.0 & 1.0 \\
1.0 & 0.0 \\
0.0 & 1.0
\end{bmatrix}
= \begin{bmatrix}
2.0 & 0.0 \\
0.0 & 3.0 \\
1.0 & 3.0
\end{bmatrix}
$$

### 3. 三维张量乘二维矩阵：降维 $\to$ 相乘 $\to$ 升维

在实际代码中，左侧输入通常是带批次维度 `B` 的三维张量 **`[B, M, K]`**，而右侧权重仍是二维矩阵 **`[K, N]`**。由于底层 `matmul` 要求左右张量阶数一致，标准做法分为三步：**先降维（3D 拍扁为 2D） $\to$ 二维矩阵相乘 $\to$ 再升维（2D 还原为 3D）**。

我们以 **2 个班级（`B=2`）、每班 2 名学生（`M=2`）、每人 3 门课成绩（`K=3`）** 的三维张量 `x_3d`（形状 `[2, 2, 3]`），乘以二维计分矩阵 $\mathbf{W}_{\text{2d}} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 1 & 0 \end{bmatrix}$（形状 `[3, 2]`，第 0 列算三科总分，第 1 列取第一科成绩）为例：

**第一步：降维**（`3D -> 2D`）

调用 `x_3d.reshape((2 * 2, 3))?`，拆掉隔开两个班级的中间那层方括号，将 2 张 $2 \times 3$ 班级小表上下拼接为 1 张 $4 \times 3$ 全校大表：

```text
原始三维 x_3d [2, 2, 3]              --->   降维后二维 x_flat_2d [4, 3]
[                                           [
  [ [1, 2, 3],   [4, 5, 6] ],    // 0 班      [1,  2,  3],   // 原 0班-学生A
  [ [10,20,30],  [40,50,60] ]    // 1 班      [4,  5,  6],   // 原 0班-学生B
]                                             [10, 20, 30],  // 原 1班-学生C
                                              [40, 50, 60]   // 原 1班-学生D
                                            ]
```

**第二步：二维矩阵相乘**（`[4, 3] × [3, 2] -> [4, 2]`）

```text
y_flat_2d = x_flat_2d [4, 3] × weight_2d [3, 2] -> 形状 [4, 2]：
[
  [6,   1],    // 学生A：总分 1+2+3=6，第一科 1
  [15,  4],    // 学生B：总分 4+5+6=15，第一科 4
  [60,  10],   // 学生C：总分 10+20+30=60，第一科 10
  [150, 40]    // 学生D：总分 40+50+60=150，第一科 40
]
```

**第三步：升维**（`2D -> 3D`）

调用 `y_flat_2d.reshape((2, 2, 2))?`，每 2 行重新套回一层班级方括号，还原为三维张量 **`[2, 2, 2]`**：

```text
升维还原后三维 y_3d [2, 2, 2]：
[
  [ [6, 1],   [15, 4] ],     // 0 班 2 名学生的汇总分
  [ [60, 10], [150, 40] ]    // 1 班 2 名学生的汇总分
]
```

封装为通用 Rust 函数仅需三行代码：

```rust
pub fn linear_3d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (b, m, k) = x.dims3()?;
    let (n, _) = w.dims2()?;
    x.reshape((b * m, k))?.matmul(&w.t()?)?.reshape((b, m, n))
}
```

```text
--- [3/6] Matrix Multiplication, Transpose & 3D <-> 2D Flatten/Unflatten ---
  2D MatMul: picked_2d [3, 4] x w^T [4, 2] -> out_2d [3, 2]: [[2.0, 0.0], [0.0, 3.0], [1.0, 3.0]]
  3D -> 2D Flatten   x_flat_2d [4, 3]: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
  2D MatMul          y_flat_2d [4, 2]: [[6.0, 1.0], [15.0, 4.0], [60.0, 10.0], [150.0, 40.0]]
  2D -> 3D Unflatten y_3d      [2, 2, 2]: [[[6.0, 1.0], [15.0, 4.0]], [[60.0, 10.0], [150.0, 40.0]]]
```

### 4. 正交矩阵、平面旋转与基变换不变性

若方阵 $\mathbf{R}$ 的各行（及各列）均为两两正交的单位向量，则称 $\mathbf{R}$ 为**正交矩阵**（Orthogonal Matrix），满足 $\mathbf{R}$ 与其转置 $\mathbf{R}^\top$ 的乘积等于**单位矩阵 $\mathbf{I}$**：

$$
\mathbf{R}\mathbf{R}^\top = \mathbf{R}^\top\mathbf{R} = \mathbf{I} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

正交矩阵在几何上代表保长保角的刚体旋转（$\|\mathbf{x}\mathbf{R}\| = \|\mathbf{x}\|$）。二维平面旋转矩阵定义为：

$$
\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

#### （1）性质一：相对旋转角的点积保角性

将二维向量 $\mathbf{q}$ 旋转 $m\theta$ 角得 $\mathbf{q}\mathbf{R}(m\theta)$，将 $\mathbf{k}$ 旋转 $n\theta$ 角得 $\mathbf{k}\mathbf{R}(n\theta)$。两者做点积时，中间的旋转矩阵合并为：

$$
\mathbf{R}(m\theta)\mathbf{R}(n\theta)^\top = \mathbf{R}\big((m - n)\theta\big)
$$

即旋转后的向量点积**仅取决于两者的旋转角度差 $(m - n)\theta$**，而与绝对位置 $m, n$ 无关。

#### （2）性质二：正交基变换不变性与“无特权基”

对于线性映射 $\mathbf{y} = \mathbf{x}\mathbf{W}$，若用任意正交矩阵 $\mathbf{R}$ 将向量旋转为 $\mathbf{x}' = \mathbf{x}\mathbf{R}$，同时将权重矩阵逆旋转为 $\mathbf{W}' = \mathbf{R}^\top\mathbf{W}$，由于中间 $\mathbf{R}\mathbf{R}^\top = \mathbf{I}$ 抵消，乘积结果保持严格不变：

$$
\mathbf{x}'\mathbf{W}' = (\mathbf{x}\mathbf{R})(\mathbf{R}^\top\mathbf{W}) = \mathbf{x}(\mathbf{R}\mathbf{R}^\top)\mathbf{W} = \mathbf{x}\mathbf{I}\mathbf{W} = \mathbf{x}\mathbf{W}
$$

取 $90^\circ$ 正交旋转矩阵 $\mathbf{R} = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$、原向量 $\mathbf{x} = [3, 4]$、原权重 $\mathbf{W} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ 验证：

- **旋转前**：$\mathbf{x}\mathbf{W} = 3 \times 1 + 4 \times 2 = \mathbf{11}$；
- **旋转后**：$\mathbf{x}' = \mathbf{x}\mathbf{R} = [-4, 3]$（各维坐标数值全部改变），$\mathbf{W}' = \mathbf{R}^\top\mathbf{W} = \begin{bmatrix} -2 \\ 1 \end{bmatrix}$；
- **新乘积**：$\mathbf{x}'\mathbf{W}' = (-4) \times (-2) + 3 \times 1 = 8 + 3 = \mathbf{11}$。

这一性质称为**正交基变换不变性**（Basis Rotation Invariance）：在纯线性通信空间中不存在**特权基**（Privileged Basis），单个坐标轴下标上的数值只是当前人为坐标系下的投影分量。

---

## 四、 矩阵乘法 vs. 逐元素乘法

| 对比项 | 矩阵乘法 (MatMul) | 逐元素乘法 (Hadamard) |
|---|---|---|
| **数学符号** | $\mathbf{M}_1 \mathbf{M}_2$ | $\mathbf{M}_1 \odot \mathbf{M}_2$ |
| **`candle` 写法** | `m1.matmul(&m2)?` | `(&m1 * &m2)?` |
| **输入形状** | `[M, K]` 与 `[K, N]` | `[M, N]` 与 `[M, N]` |
| **输出形状** | `[M, N]`（消去内侧维度 `K`） | `[M, N]`（形状不变） |
| **计算规则** | 第 $i$ 行与第 $j$ 列点积求和 | 相同坐标 $(i, j)$ 直接相乘 |

给定两个 $2 \times 2$ 矩阵 $\mathbf{M}_1 = \begin{bmatrix} 0.5 & 2.0 \\ 0.0 & 1.5 \end{bmatrix}$ 与 $\mathbf{M}_2 = \begin{bmatrix} 4.0 & 3.0 \\ 9.0 & 2.0 \end{bmatrix}$：

- **逐元素乘法 `(&m1 * &m2)?`**（对应位置直接相乘，常用于门控过滤，如左下角 `0.0` 将 `9.0` 归零）：

$$
\mathbf{M}_1 \odot \mathbf{M}_2 = \begin{bmatrix} 0.5 \times 4.0 & 2.0 \times 3.0 \\ 0.0 \times 9.0 & 1.5 \times 2.0 \end{bmatrix} = \begin{bmatrix} \mathbf{2.0} & \mathbf{6.0} \\ \mathbf{0.0} & \mathbf{3.0} \end{bmatrix}
$$

- **矩阵乘法 `m1.matmul(&m2)?`**（行与列做点积求和）：

$$
\mathbf{M}_1 \mathbf{M}_2 = \begin{bmatrix} 0.5\times 4.0 + 2.0\times 9.0 & 0.5\times 3.0 + 2.0\times 2.0 \\ 0.0\times 4.0 + 1.5\times 9.0 & 0.0\times 3.0 + 1.5\times 2.0 \end{bmatrix} = \begin{bmatrix} \mathbf{20.0} & \mathbf{5.5} \\ \mathbf{13.5} & \mathbf{3.0} \end{bmatrix}
$$

---

## 五、 维度重塑、交换与内存连续性

在多维张量代码中，最常见的一组链式操作是：

```rust
let reshaped = scores_2d.reshape((3, 2, 2))?;
let transposed = reshaped.transpose(0, 1)?;
let contiguous = transposed.contiguous()?;
```

我们用一张 **3 名学生、每人 4 门课** 的二维成绩单 `scores_2d`（形状 `[3, 4]`）演示每一步的数值变化（十位数代表学生 `1..3`，个位数代表课程 `1..4`；前 2 门为文科组，后 2 门为理科组）：

```text
原始二维 scores_2d（形状 [3, 4]：3 名学生，每人 4 门课）：
[
  [11, 12, 13, 14],   // 学生 1
  [21, 22, 23, 24],   // 学生 2
  [31, 32, 33, 34]    // 学生 3
]
```

### 1. 拆分维度（`reshape`）

`reshape` 在**不改变内存元素顺序、保持元素总数相等**的前提下，重新为数据划分括号层级。

由于 $3 \times 4 = 3 \times 2 \times 2 = 12$，调用 `scores_2d.reshape((3, 2, 2))?` 将每人 4 门课按“**2 个文理组、每组 2 门课**”套上一层括号：

```text
reshape((3, 2, 2)) 后（形状：[3 名学生, 2 个文理组, 每组 2 门课]）：
[
  [ [11, 12], [13, 14] ],   // 学生 1：[文科组, 理科组]
  [ [21, 22], [23, 24] ],   // 学生 2：[文科组, 理科组]
  [ [31, 32], [33, 34] ]    // 学生 3：[文科组, 理科组]
]
```

### 2. 交换维度（`transpose`）

此时第 0 维是“学生”，第 1 维是“文理组”。若文科老师与理科老师需要各自拿到全班的文科表与理科表独立算分，就需要调用 `.transpose(0, 1)?` **对调第 0 维与第 1 维的归档层级**，将形状变为 **`[2, 3, 2]`**（`[2 个文理组, 3 名学生, 每组 2 门课]`）：

```text
transpose(0, 1) 后（形状：[2 个文理组, 3 名学生, 每组 2 门课]）：
[
  [ [11, 12], [21, 22], [31, 32] ],   // 第 0 组（全班 3 人的文科成绩表 [3, 2]）
  [ [13, 14], [23, 24], [33, 34] ]    // 第 1 组（全班 3 人的理科成绩表 [3, 2]）
]
```

### 3. 整理物理内存连续性（`contiguous`）

计算机物理内存是一维线性的。初始创建 `scores_2d` 时，内存顺序为 `[11, 12, 13, 14, 21, 22, ...]`：

- **调用 `reshape` 后**：逻辑顺序与物理内存顺序一致，属于**连续内存**（`is_contiguous = true`）；
- **调用 `transpose(0, 1)` 后**：为节省开销，`transpose` 仅修改了维度步长（Stride）元数据而未真正搬移内存；此时逻辑上要求先读全班文科 `[11, 12, 21, 22, 31, 32]`，物理内存里却隔着理科数据需要跳着读，变成**非连续内存**（`is_contiguous = false`）；
- **调用 `.contiguous()?` 后**：按转置后的新逻辑顺序在物理内存中重新排齐为 `[11, 12, 21, 22, 31, 32, 13, 14, 23, 24, 33, 34]`，恢复 `is_contiguous = true`，以满足后续底层矩阵算子的连续内存要求。

### 4. 维度插入与广播复制（`unsqueeze` & `expand`）

若要将 `[2, 3, 2]` 中的 2 个大组各自连续复制 3 份变为 **`[6, 3, 2]`**，由于 `.expand` 只能将长度为 `1` 的维度放大为 $N$ 份，标准写法分为三步：

1. **插空位 `unsqueeze(1)`**：在维度 1 插入长度为 1 的新维度，`[2, 3, 2] -> [2, 1, 3, 2]`；
2. **原地复制 `expand((2, 3, 3, 2))`**：将长度为 1 的维度放大为 3 份，`[2, 1, 3, 2] -> [2, 3, 3, 2]`；
3. **合并维度 `reshape((6, 3, 2))`**：将前两维 $2 \times 3$ 合并为 6，`[2, 3, 3, 2] -> [6, 3, 2]`。

```text
--- [5/6] Reshape, Transpose & Contiguous (3 Students x 4 Courses) ---
  1. Original [3, 4] (3 students, 4 courses):
     [[11.0, 12.0, 13.0, 14.0], [21.0, 22.0, 23.0, 24.0], [31.0, 32.0, 33.0, 34.0]]
  2. After reshape(3, 2, 2) [3 students, 2 groups, 2 courses] (is_contiguous = true):
     [[[11.0, 12.0], [13.0, 14.0]], [[21.0, 22.0], [23.0, 24.0]], [[31.0, 32.0], [33.0, 34.0]]]
  3. After transpose(0, 1)  [2 groups, 3 students, 2 courses] (is_contiguous = false):
     Group 0 (Arts    for all 3 students): [[11.0, 12.0], [21.0, 22.0], [31.0, 32.0]]
     Group 1 (Science for all 3 students): [[13.0, 14.0], [23.0, 24.0], [33.0, 34.0]]
  4. After .contiguous()    [2, 3, 2] (is_contiguous = true):
     Flattened physical order: [11.0, 12.0, 21.0, 22.0, 31.0, 32.0, 13.0, 14.0, 23.0, 24.0, 33.0, 34.0]
  5. Broadcast expand: [2, 3, 2] (2 groups) each repeated 3x -> [6, 3, 2] (6 groups)
```

---

## 六、 RMSNorm、Softmax 与 SiLU

### 1. 均方根归一化（RMSNorm）

在深层网络中，向量连续乘多层矩阵后数值幅度极易急剧膨胀或衰减。**均方根归一化**（Root Mean Square Normalization） 的作用是：保留向量内部各元素的相对比例不变，将其整体幅度拉回标准范围。

#### （1）为什么用均方根（RMS），不用算术平均或最大绝对值？

对比三种候选方案即可看清 RMS 的数学优势：

- **为什么不能除以算术平均值 $\frac{1}{d}\sum v_i$？**
  正负数会互相抵消。例如向量 `[-5, 5, -5, 5]` 的算术平均值为 `0`，会导致除零崩溃。
- **为什么不能除以最大绝对值 $\max |v_i|$？**
  对极端离群噪点（Outlier）过于敏感。假设一个 1000 维向量中有 999 个正常元素均为 `1.0`，仅 1 个噪点元素波动到 `100.0`：若除以最大值 `100.0`，其余 999 个正常元素会全部萎缩为 `0.01`；而采用均方根平摊后，$\text{RMS} = \sqrt{(999 \times 1^2 + 100^2)/1000} \approx \mathbf{3.32}$，其余 999 个元素除以 `3.32` 后仍保持 `0.30` 的有效幅度。
- **为什么不用绝对值平均 $\frac{1}{d}\sum |v_i|$？**
  平方和开根号直接对应欧氏空间向量的几何模长 $\|\mathbf{v}\|_2 / \sqrt{d}$（各向同性），且平方函数 $x^2$ 在 $0$ 点处处光滑可导，无绝对值 $|x|$ 的尖锐折角。

因此，RMSNorm 先求各元素平方的均值并开根号（加极小常数 $\epsilon = 10^{-5}$ 防全零除零），再逐元素除以均方根并乘可学习缩放向量 $\boldsymbol{\gamma}$：

$$
\text{RMS}(\mathbf{v}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} v_i^2 + \epsilon}, \qquad \operatorname{RMSNorm}(v_i) = \frac{v_i}{\text{RMS}(\mathbf{v})} \cdot \gamma_i
$$

#### （2）手算对比验证

比较比例相同但幅度相差 100 倍的两个 4 维向量 $\mathbf{v}_{\text{normal}} = [2, -4, 4, 8]$ 与 $\mathbf{v}_{\text{huge}} = [200, -400, 400, 800]$（设 $\gamma_i = 1$）：

- **$\mathbf{v}_{\text{normal}}$**：平方平均为 $(4 + 16 + 16 + 64)/4 = 25$，均方根 $\text{RMS} = \sqrt{25} = \mathbf{5}$，除以 $5$ 得 **`[0.4, -0.8, 0.8, 1.6]`**；
- **$\mathbf{v}_{\text{huge}}$**：平方平均为 $1000000/4 = 250000$，均方根 $\text{RMS} = \sqrt{250000} = \mathbf{500}$，除以 $500$ 同样得到 **`[0.4, -0.8, 0.8, 1.6]`**（且归一化后新向量的均方根恒等于 `1.0`）。

### 2. Softmax 概率归一化

给定任意实数得分向量 $[s_1, s_2, \dots, s_K]$，**Softmax** 先对每个得分取自然指数 $e^{s_i}$（消除负号并拉开高分差距，且满足 $e^{-\infty} = 0$），再除以指数总和，使其转化为各项非负且总和恒为 $1.0$ 的概率分布：

$$
\operatorname{Softmax}(s_i) = \frac{e^{s_i}}{\sum_{j=1}^{K} e^{s_j}}
$$

- **普通得分 `[1.0, 2.0, 3.0]`**：取指数得 $[2.72, 7.39, 20.09]$（总和 $30.20$），除以总和得 **`[0.09, 0.24, 0.67]`**；
- **带负无穷掩码 `[1.0, 1.0, -inf]`**：取指数得 $[2.72, 2.72, 0.0]$，除以总和得 **`[0.5, 0.5, 0.0]`**（`-inf` 项权重严格归零）。

```text
--- [6/6] RMS Normalization & Softmax ---
  v_normal [2, -4, 4, 8]         (RMS = 5)   -> RMSNorm: [0.39999992, -0.79999983, 0.79999983, 1.5999997]
  v_huge   [200, -400, 400, 800] (RMS = 500) -> RMSNorm: [0.4, -0.8, 0.8, 1.6]
  Scores [1.0, 2.0, 3.0]  -> Softmax: [0.09003057, 0.24472848, 0.66524094]
  Scores [1.0, 1.0, -inf] -> Softmax: [0.5, 0.5, 0.0]
```

### 3. SiLU（Swish）非线性激活函数

若两层矩阵乘法之间不加非线性函数，由结合律 $\mathbf{x}\mathbf{W}_1\mathbf{W}_2 = \mathbf{x}(\mathbf{W}_1\mathbf{W}_2)$，多层网络会坍缩为单层线性变换。

现代模型广泛采用 **SiLU**（Sigmoid Linear Unit） 逐元素非线性函数，定义为输入 $x$ 乘以其自身的 Sigmoid 门控值 $\sigma(x) = \frac{1}{1 + e^{-x}}$：

$$
\operatorname{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

- 当 $x = -10.0$ 时：$\sigma(-10) \approx 0.000045$，$\operatorname{SiLU}(-10.0) \approx \mathbf{-0.00045}$（负值被平滑抑制至接近 $0$）；
- 当 $x = 0.0$ 时：$\operatorname{SiLU}(0.0) = \mathbf{0.0}$；
- 当 $x = 2.0$ 时：$\sigma(2.0) \approx 0.8808$，$\operatorname{SiLU}(2.0) \approx \mathbf{1.7616}$（正值近似按原比例放行）。

---

## 七、 Rust 项目结构

一个标准 Rust 代码包（Crate）的目录结构如下：

```text
my-project/
├── Cargo.toml       # 项目清单文件（管理包版本与依赖库，类似 pyproject.toml）
└── src/
    ├── lib.rs       # 核心库入口：声明并导出功能模块（pub mod xxx;），包含单元测试
    └── main.rs      # 可执行程序入口：包含 fn main()，解析命令行并调用 lib.rs 模块
```

- **`Cargo.toml`**：运行 `cargo run` 或 `cargo test` 时，`cargo` 会自动下载并编译其中声明的依赖；
- **`pub mod` 与 `use`**：在 `src/lib.rs` 中声明 `pub mod ops;` 即可挂载 `src/ops.rs` 文件，其他模块通过 `use crate::ops::函数名;` 引入公开函数。

---

## 八、 变量绑定与基础类型

### 1. 默认只读与可变绑定（`let` vs `let mut`）

Rust 中使用 `let` 声明的变量默认**不可变**（Immutable）；只有显式加上 `mut` 关键字后才允许修改：

```rust
let rows = 3;
// rows = 4;     // 编译报错：默认只读

let mut sum = 0; // 加上 mut 后允许后续更新
sum = sum + 10;
```

### 2. 四种基础数值类型与元组解构

| 类型 | 位宽与类别 | 典型用途 |
|---|---|---|
| **`usize`** | 64 位无符号整数（随 CPU 字长） | 数组长度、张量维度、循环下标 |
| **`u32`** | 32 位无符号整数 | 词表 Token ID、行号索引 |
| **`f32`** | 32 位单精度浮点数 | 张量内部的标准数值计算 |
| **`f64`** | 64 位双精度浮点数 | 高精度标量参数（如 `1e-5`） |

数值类型转换统一使用 **`as` 关键字**显式转换（如 `(d_h as f64).sqrt()`）；函数返回多值元组时可用 `let (b, t, d) = tensor.dims3()?;` 一行解构。

---

## 九、 所有权与借用（`&` 与 `&mut`）

在 Rust 张量代码中，变量名前常带有取引用符号 **`&`**：

```rust
let elem_wise = (&m1 * &m2)?;
let mat_mul = m1.matmul(&m2)?;
```

这源于 Rust 的所有权与借用规则，向函数传递变量共有三种方式：

- **直接传值 `m1`**（移交所有权 Move）：把 `m1` 的所有权彻底交给对方，调用结束后 `m1` 立即销毁失效，下一行无法再使用；
- **传不可变引用 `&m1`**（只读借用 Shared Borrow）：仅将 `m1` 的只读内存地址借给算子读取，既不复制底层矩阵内存，算完也不销毁 `m1`，下一行仍可继续使用；
- **传可变引用 `&mut kv_caches`**（可写借用 Mutable Borrow）：允许被调用函数原地修改或向该容器追加新数据。

---

## 十、 结构体、枚举与模式匹配

### 1. 结构体与方法（`struct` & `impl`）

`struct` 用于定义数据结构，`impl` 用于挂载方法（函数最后一行不加分号 `;` 即作为返回值）：

```rust
#[derive(Debug, Clone)]
pub struct MatrixConfig {
    pub rows: usize,
    pub cols: usize,
}

impl MatrixConfig {
    pub fn total_elements(&self) -> usize {
        self.rows * self.cols
    }
}
```

### 2. 枚举与空值安全（`enum`、`match` 与 `Option<T>`）

Rust 语言没有 `null`。当一个槽位可能存放数据 `T`、也可能为空时，强制使用标准库枚举 **`Option<T>`**：

- **`Some(v)`**：存有有效数据 `v`；
- **`None`**：当前为空。

读取 `Option<T>` 时，编译器强制要求用 `match` 同时处理 `Some` 与 `None` 两个分支，从根源上杜绝空指针崩溃：

```rust
let updated = match cache_slot.take() {
    Some(prev) => Tensor::cat(&[&prev, &new_tensor], 2)?, // 已有历史缓存：拼接
    None => new_tensor,                                   // 首次为空：直接存入
};
```

---

## 十一、 错误处理（`Result` 与 `?`）

张量操作可能因维度不匹配或文件缺失而报错。Rust 不使用异常捕获，而是令可能失败的函数返回 **`Result<T>`** 枚举（成功为 `Ok(val)`，失败为 `Err(e)`）。

在调用末尾加上问号 **`?`** 是错误传播语法糖：

```rust
let w_t = w.t()?;
let out_2d = picked_2d.matmul(&w_t)?;
```

- 若返回 `Ok(val)`，`?` 自动拆出内部的 `Tensor` 赋给左侧变量；
- 若返回 `Err(e)`，`?` 立即中止当前函数并将错误向上返回给调用者。

---

## 十二、 数组、切片、迭代器与算子速查表

### 1. 定长数组、动态数组与切片

- **定长数组 `[1.0f32, 2.0, 3.0]`**：编译期长度固定（类型 `[f32; 3]`）。首个元素标注 `1.0f32` 后，同数组后续元素会自动推导为 `f32`。
- **动态数组 `Vec<T>`**：运行期可动态增长。注意 `vec![...]` 中逗号与分号的区别：

```rust
let list_a: Vec<f32> = vec![1.0f32, 2.0, 3.0]; // 逗号：列举 3 个具体元素
let zeros: Vec<f32>  = vec![0.0f32; 10];       // 分号：[初始值; 重复次数]，生成 10 个 0.0
```

- **只读切片 `&[T]`**：在定长数组或 `Vec` 前加 `&`（如 `&[1.0f32, 2.0, 3.0]`），即得到不复制内存的只读视图（内部仅含起始指针与长度），可直接传给 `Tensor::from_slice(&data, shape, dev)?`。

### 2. 左闭右开区间 `0..n` 与迭代器流水线 `.map().collect()`

`0..4` 等价于 Python 的 `range(0, 4)`，按序产出整数 `0, 1, 2, 3`。当需要按公式批量生成数组时，Rust 常将 5 行 `for` 循环简写为一行三段式迭代器流水线：

```rust
let even_floats: Vec<f32> = (0..4).map(|i| (i * 2) as f32).collect();
```

```text
  第 1 站：数据源              第 2 站：映射转换闭包               第 3 站：收集装箱
     (0..4)         --->   .map(|i| (i * 2) as f32)    --->     .collect()
  产出: 0, 1, 2, 3          变为: 0.0, 2.0, 4.0, 6.0         装入: Vec<f32> 数组
```

- **第 1 站 `(0..4)`**：依次产出整数 `0, 1, 2, 3`；
- **第 2 站 `.map(|i| ...)`**：两条竖线 `|i|` 是 Rust 的**匿名函数（闭包）参数声明**（等价于 Python 的 `lambda i: ...` 或 JS 的 `(i) => ...`），将每个整数 `i` 映射为浮点数 `(i * 2) as f32`；
- **第 3 站 `.collect()`**：启动惰性迭代器并将结果打包进左侧声明的 `Vec<f32>` 容器中（若将 `.collect()` 换为 `.sum()` 则直接对所有元素求和；若对已有数组启动流水线则先调 `.iter()`）。

### 3. candle-core 算子速查表

| 操作功能 | PyTorch 写法 | `candle-core` 写法 |
|---|---|---|
| **从数组创建** | `torch.tensor(data).reshape(5, 4)` | `Tensor::from_slice(&data, (5, 4), dev)?` |
| **全 0 / 全 1** | `torch.zeros((2, 3))` | `Tensor::zeros((2, 3), DType::F32, dev)?` |
| **查看形状** | `x.shape` | `x.dims()` / `x.dims3()?` |
| **类型转换** | `x.to(torch.float32)` | `x.to_dtype(DType::F32)?` |
| **按行号检索** | `table[row_indices]` | `table.index_select(&row_indices, 0)?` |
| **重塑形状** | `x.reshape(1, 3, 4)` | `x.reshape((1, 3, 4))?` |
| **展平为一维** | `x.flatten()` | `x.flatten_all()?` |
| **二维转置** | `w.T` | `w.t()?` |
| **交换指定维** | `x.transpose(1, 2)` | `x.transpose(1, 2)?` |
| **内存重排齐** | `x.contiguous()` | `x.contiguous()?` |
| **矩阵乘法** | `a @ b` | `a.matmul(&b)?` |
| **逐元素加乘** | `a + b` / `a * b` | `(&a + &b)?` / `(&a * &b)?` |
| **切片子区间** | `x[offset : offset + len]` | `x.narrow(0, offset, len)?` |
| **按下标索引** | `x[0, idx]` | `x.i((0, idx))?` |
| **沿维度拼接** | `torch.cat([a, b], dim=2)` | `Tensor::cat(&[&a, &b], 2)?` |
| **最大值下标** | `v.argmax(dim=0).item()` | `v.argmax(0)?.to_scalar::<u32>()?` |
| **导出为数组** | `x.tolist()` | `x.to_vec1::<f32>()?` / `x.to_vec2::<f32>()?` |

> **补充说明**：使用 `.i(...)` 按下标索引前，需在文件顶部引入 `use candle_core::IndexOp;`；`dims3()?` 与 `dims4()?` 可直接将三维或四维张量的形状解构为 Rust 元组 `(d0, d1, d2)`。

---

准备好以上 **6 条线性代数法则** 与 **6 个 Rust 核心语法** 后，接下来即可正式进入 **[《第零篇｜大模型推理全流程：从文本到下一个 Token》](../ep00-inference-journey/draft.md)**。
