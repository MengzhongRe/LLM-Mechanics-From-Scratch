# [Paper Read] Large Language Models are Zero-Shot Reasoners

- **Authors**: Takeshi Kojima, Shixiang Shane Gu, et al. (University of Tokyo, Google Research)
- **Year**: 2022 (NeurIPS)
- **Link**: [arXiv:2205.11916](https://arxiv.org/abs/2205.11916)
- **Tags**: #Zero-Shot-CoT #Prompt-Engineering #Reasoning #System2

---

## 📖 1. 核心背景与颠覆性假设 (Introduction)

### 1.1 时代背景
在 2022 年初，Jason Wei 等人发表了 *Chain-of-Thought (CoT)* 论文，确立了一个范式：**要想让大模型做逻辑推理，必须使用 Few-Shot Prompting。** 即：用户必须手写几个 `Q -> Reasoning -> A` 的高质量示例（Demonstrations）喂给模型，模型才能学会模仿这种推理路径。

### 1.2 本文的颠覆 (The Paradigm Shift)
本文作者（Kojima 等人）挑战了上述共识。他们提出了一个极具哲学意味的假设：
> **Hypothesis:** 逻辑推理能力不是通过 Few-Shot "学会" (Learned) 的，而是模型在预训练海量数据后**先天具备** (Inherent) 的。它潜伏在模型的 **Latent Space（潜空间）** 中。

我们不需要“教”模型如何推理，只需要找到一把正确的**“钥匙”**（Trigger Sentence）来**唤醒**这种能力。

---

## 🔬 2. 核心方法：两阶段提示法 (Methodology: Two-Stage Prompting)

为了在零样本（Zero-Shot）下实现推理，作者设计了一个符合逻辑推导过程的**两阶段管道（Pipeline）**。这在逻辑上对应了 **“展开推导”** 与 **“归纳结题”** 的过程。

### Stage 1: 推理生成 (Reasoning Extraction)
*   **输入 (Input):** 问题 $x$ + 咒语模板 $T$。
    *   *The Magic Spell:* **"Let's think step by step." (让我们一步步思考)**
*   **输出 (Output):** 模型不再直接输出答案，而是生成了一段长文本 $z$（推理路径/思维链）。
*   **逻辑原理:** 这句提示词改变了模型的**生成概率分布**。它抑制了直接映射到答案的 System 1 直觉路径，激活了按时序展开的 System 2 逻辑路径。

### Stage 2: 答案提取 (Answer Extraction)
*   **输入 (Input):** 原问题 $x$ + 推理路径 $z$ + 归结提示词 $A$。
    *   *The Extraction Spell:* **"Therefore, the answer is" (因此，答案是)**
*   **输出 (Output):** 最终答案 $\hat{y}$。
*   **逻辑原理:** 这是一个**逻辑归约（Reduction）**操作。它强迫模型从发散的思维链中收敛，将推导过程坍缩为一个确定的真值或数值。

---

## 🧪 3. 提示词语义学与消融实验 (Prompt Engineering Analysis)

作者在 Section 4 中进行了一场精彩的“咒语海选”。这实际上是对模型如何理解人类指令的**语义学分析**。

| 提示词 (Prompt Template) | 准确率 (Accuracy on MultiArith) | 逻辑学深度解析 |
| :--- | :--- | :--- |
| **"Let's think step by step"** | **78.7% (SOTA)** | **最优解。** 显式强制了**线性时序 (Linearity)** 和 **步骤化 (Algorithm)**。模型被限制在“每一步只做一件事”的模式中。 |
| "Let's think about this logically" | 74.5% | 强调了“理性”，但缺乏对“步骤”的强制约束，导致推理结构可能混乱。 |
| "Let's solve this problem" | 65.3% | 过于宽泛 (Generic)。没有触发特定的逻辑子空间。 |
| "Let's be realistic" | 47.3% | 引入了无关的语用约束。 |
| "The answer is" | 17.7% (Baseline) | **灾难性失败。** 强制模型直接收敛，跳过了所有中间推导。这是标准 Zero-Shot 的本质。 |

**关键结论:**
Prompt 不仅仅是输入信息，它是**功能选择器 (Function Selector)**。不同的自然语言指令激活了神经网络中完全不同的计算回路。

---

## 📊 4. 实验结果 (Quantitative Experiments)

实验覆盖了 12 个数据集，包括算术、常识推理和符号推理。

1.  **巨大提升 (Massive Gains):**
    *   在 MultiArith 上，从 **17.7%** (Standard Zero-Shot) 暴涨至 **78.7%** (Zero-Shot-CoT)。
    *   在 GSM8K (小学奥数) 上，从 10.4% 提升至 40.7%。
2.  **比肩 Few-Shot:**
    *   令人震惊的是，一句 Prompt 的效果竟然超过了 Standard Few-Shot（给例子但不给推理），并且在某些任务上逼近了 Wei et al. 精心设计的 8-shot CoT。
    *   *Meaning:* **形式（推理结构）的价值 > 内容（具体样本的价值）。**
3.  **符号推理 (Symbolic Reasoning):**
    *   在 "Coin Flip"（抛硬币）和 "Last Letter Concatenation" 任务中，模型完美模拟了**状态机 (State Machine)** 的运作，能够追踪变量在每一步的变化。

---

## 🧠 5. 深度讨论与定性分析 (Section 5 Analysis)

### 5.1 规模效应 (Scaling Law)
Zero-Shot CoT 表现出极强的**涌现性 (Emergence)**：
*   **Small Models (<10B):** "Let's think step by step" **无效**，甚至导致性能下降。小模型无法维持长链条的一致性，生成的步骤往往是幻觉。
*   **Large Models (100B+):** 只有在 GPT-3 (175B) 和 InstructGPT 这个级别，推理能力才被成功唤醒。
*   *逻辑推论:* 逻辑不是教出来的，是**规模堆出来的**；提示词只是去取用它。

### 5.2 错误类型 (Error Analysis)
尽管效果显著，但 Zero-Shot CoT 仍存在以下缺陷（这也是后续研究的改进点）：
1.  **Hallucination (幻觉):** 推理过程逻辑通顺，但甚至前提条件（Premise）都是捏造的。
2.  **Calculation Error (计算错误):** 步骤对，公式对，但算术运算（Arithmetic Operation）算错了。LLM 依然不是计算器。
3.  **Semantic Misunderstanding (语义误解):** 也就是“审题错误”。

### 5.3 Instruct Tuning 的作用
实验发现，**InstructGPT (Davinci-002)** 的效果远好于原始 GPT-3。
*   这是因为 Instruct Tuning（指令微调）让模型具备了**“遵循命令”**的能力。原始模型可能把 "Let's think step by step" 当作小说续写，而 Instruct 模型将其视为必须执行的算法指令。

---

## 📝 6. 核心总结与逻辑学启示 (My Key Takeaways)

作为逻辑学背景的研究者，这篇论文揭示了 AI 的三个本质特征：

1.  **Latent Reasoning (隐性逻辑):**
    推理能力不需要像 Few-Shot 那样通过“外部注入”，它是大模型内部**先验存在**的。这挑战了经验主义的学习观点。
2.  **The Bridge of Thought (思维之桥):**
    $X \to Y$ (直接预测) 是困难的，甚至是不可计算的。
    $X \to Z \to Y$ (引入中间变量 $Z$) 让问题变得可解。
    **中间过程 $Z$ (Reasoning Path) 本身就是智能的载体**。
3.  **Linearity constraint (线性约束):**
    "Step by step" 之所以有效，是因为它强制模型将多维的向量计算，投影到了**一维的线性时序逻辑**上，这符合人类形式逻辑（演绎推理）的本质。

## 🔗 下一步阅读计划 (Next Steps)
虽然 Zero-Shot CoT 很强，但它是一个**贪婪解码 (Greedy Decoding)** 过程——只生成一条路，走错了就全错了。
*   *Idea:* 既然模型是概率性的，能不能让它生成 10 条推理路径，然后投票选出得票最多的答案？
*   *Next Paper:* **Self-Consistency Improves Chain of Thought Reasoning (2022)**。这将把推理从“单点逻辑”提升到“集成逻辑”。