# [Paper Read] QLoRA: Efficient Finetuning of Quantized LLMs

- **Authors**: Tim Dettmers (University of Washington), et al.
- **Year**: 2023 (NeurIPS)
- **Link**: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Tags**: #Quantization #NF4 #Double-Quantization #System-Optimization #Memory-Efficient

---

## 1. 核心痛点 (The Problem)

微调大模型（如 65B）的主要瓶颈在于**显存 (GPU Memory)**。
- 全量微调需要加载：**模型权重 (16-bit) + 梯度 + 优化器状态**。
- LoRA 虽然减少了梯度和优化器状态，但**基座模型 (Base Model)** 依然要以 16-bit 驻留显存。对于 5070 Ti (16G) 这样的消费级显卡，连加载一个 14B 模型都很困难。

**QLoRA 的目标：** 在**不损失性能**的前提下，将基座模型压缩到 **4-bit**，从而在消费级显卡上实现大模型微调。

## 2. 核心技术创新 (The Trinity of Innovations)

### 2.1 4-bit NormalFloat (NF4) —— 信息论最优量化
传统的 Int4 量化假设数据是均匀分布的（线性），但神经网络权重服从**正态分布 $\mathcal{N}(0, 1)$**。NF4 是一种基于**分位数 (Quantile)** 的非线性量化数据类型。

*   **构造逻辑：**
    1.  **概率切分：** 将正态分布的累积概率密度 (CDF) 切分为面积相等的 16 份。
    2.  **逆映射：** 使用逆 CDF 函数 ($F^{-1}$) 计算每份的中心点，得到 16 个代表元。
    3.  **非对称修正：** 强行在正负区间之间插入精确的 **0.0**（对于稀疏性至关重要）。
*   **Codebook (码本)：** 这 16 个代表元构成了通用的查找表。因为所有模型的权重都近似正态分布，所以**这张表是通用的、固定的**。
*   **价值：** 在 0 附近（数据密集区）刻度极密，保留了最大信息量。

### 2.2 双重量化 (Double Quantization) —— 极致压缩
为了减少量化误差，QLoRA 使用了 **Block-wise Quantization**（每 64 个参数共用一个缩放因子 $S$）。
- **问题：** 这些缩放因子 $S$ 本身是 FP32/BF16 的，虽然占比小，但在大模型下依然可观（约 0.5 bit/param）。
- **解法：** 对这些缩放因子 $S$ **再进行一次量化**（FP32 $\to$ FP8）。
- **收益：** 平均每参数节省 0.37 bit。对于 65B 模型，这能额外省下 3GB 显存。

### 2.3 分页优化器 (Paged Optimizers) —— 防 OOM 机制
利用 NVIDIA 的统一内存特性，在显存出现峰值（Spike）时，自动将优化器状态（Optimizer States）**逐页 (Page-by-Page)** 转移到 CPU 内存中，计算时再取回。防止训练中途崩溃。

## 3. 计算流与解压原理 (De-quantization Workflow)

QLoRA 的本质是 **“存储换算力” (Compute for Memory)**。

- **静态存储 (Storage):** 显存中存储的是 **NF4 索引 (Indices)**。
- **动态计算 (Computation):**
    1.  数据流经某一层。
    2.  **即时解压 (On-the-fly Dequantization):** 硬件通过查表和乘法，将 NF4 索引映射回 BF16 真值。
        $$ W_{\text{BF16}} = \text{Codebook}[\text{Index}_{\text{Int4}}] \times S_{\text{Block}} $$
    3.  **矩阵乘法:** 在 Tensor Core 中使用高精度的 **BF16** 进行 $X \cdot W$ 运算。
    4.  **释放:** 运算结束后，立即释放 BF16 临时变量，显存回落。

## 4. 训练逻辑 (Training Dynamics)

- **基座模型 ($W_{\text{base}}$):** 冻结 (Frozen)，以 NF4 格式存储。**不更新参数，不计算梯度**。
- **适配器 ($W_{\text{LoRA}}$):** 可训练 (Trainable)，以 BF16 格式存储。**计算梯度并更新**。
- **精度混合:**
    - 前向/后向传播经过基座时，权重临时解压为 BF16。
    - 梯度的“流动”经过基座，但最终只“沉淀”在 LoRA 的参数矩阵里。

---

## 💡 逻辑学与工程视角 (My Takeaways)

**1. 形式与质料 (Form and Matter)**
- **NF4 Codebook** 是“形式” (Form)，代表了正态分布的普遍规律（共相）。
- **Indices 和 Scales** 是“质料” (Matter)，代表了特定模型的具体知识（殊相）。
- QLoRA 证明了我们只需要存储“质料”的索引，就能重构出高精度的知识表达。

**2. 信息的有损与无损**
- 从 BF16 到 NF4 是**有损压缩**（丢失了尾数精度）。
- 但从模型效果来看，由于神经网络的过参数化和鲁棒性，这种微小的数值损失在逻辑功能上是**近似无损**的。

**3. 5070 Ti 的解放**
QLoRA 是消费级显卡进行大模型科研的**入场券**。它打破了显存的物理限制，让个人开发者能够触碰 SFT 的核心。
