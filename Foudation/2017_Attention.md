# [Paper Read] Attention Is All You Need (Transformer)

- **Authors**: Google Brain (Vaswani et al.)
- **Year**: 2017 (NeurIPS)
- **Link**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Tags**: #Transformer #Self-Attention #Base #SOTA

---

## 1. 核心思想 (The Big Idea)

这篇论文推翻了 RNN/LSTM 在序列建模中的统治地位。
它提出：**处理序列数据不需要循环（Recurrence）和卷积（Convolution），只需要注意力机制（Attention）。**

**工程意义：**
- **并行化 (Parallelization):** RNN 必须 $t \rightarrow t+1$ 串行计算，Transformer 可以一次性并行处理所有 Token，极大释放了 GPU 算力。
- **长距离依赖 (Long-term Dependency):** 任意两个词之间的距离都是 1 (通过 Attention 矩阵直接相连)，解决了 LSTM 的梯度消失问题。

## 2. 关键技术 (Technical Details)

### 2.1 Scaled Dot-Product Attention
这是 Transformer 的原子操作。公式如下：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

- **Q (Query):** 查询向量。
- **K (Key):** 键向量（索引）。
- **V (Value):** 值向量（内容）。
- **$\sqrt{d_k}$ (Scaling):** 为什么要除以根号维数？
    - 为了防止点积结果过大，导致 Softmax 进入饱和区（梯度接近 0），从而导致**梯度消失**。

### 2.2 Multi-Head Attention (多头注意力)
将 $d_{model}$ 切分为 $h$ 个头（Heads）。
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $$
- 允许模型在不同的**表示子空间 (Representation Subspaces)** 关注不同的信息（例如：一个头关注语法结构，一个头关注语义指代）。

### 2.3 Positional Encoding (位置编码)
由于没有 RNN 的时间步概念，必须显式注入位置信息。
$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
- 使用正弦/余弦函数的叠加，使得模型能够学习到相对位置关系。

### 2.4 Architecture (Encoder-Decoder)
- **Encoder:** 6层，用于理解输入（双向可见）。
- **Decoder:** 6层，用于生成输出（Masked，只能看左边）。

---

## 🧠 Logic & Philosophy Perspective (逻辑学视角)

**1. 模糊检索系统 (Fuzzy Retrieval System)**
Attention 机制本质上是一个**软逻辑（Soft Logic）**检索系统。
- 传统的数据库查询是 Hard Look-up (Key 匹配就返回 Value，否则为 NULL)。
- Attention 是计算 Query 与 Key 的**相似度（Similarity）**，将其作为权重，对 Value 进行加权求和。这是一种**基于向量空间的加权推理**。

**2. 全局互联 (Global Interconnectivity)**
在逻辑图中，Transformer 相当于构建了一个**完全图 (Complete Graph)**，每个 Token 都直接与其他所有 Token 相连。Softmax 决定了这些连接的**强度 (Weights)**。

---

## 💡 Interview QA (面试必问)

- **Q: 为什么要用 LayerNorm 而不是 BatchNorm？**
  - A: BN 对 Batch Size 敏感，且处理变长序列（NLP常见）效果差；LN 对每个样本独立归一化，更适合序列任务。
- **Q: Decoder 为什么要 Mask？**
  - A: 保持**自回归 (Auto-regressive)** 属性。在预测 $t$ 时，不能偷看 $t+1$ 的信息（防止信息泄露）。