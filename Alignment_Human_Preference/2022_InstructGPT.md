# [Paper Read] Training language models to follow instructions with human feedback (InstructGPT) - Part I: Methodology

- **Authors**: Long Ouyang, Jeff Wu, Xu Jiang, et al. (OpenAI)
- **Year**: 2022 (NeurIPS)
- **Link**: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Focus**: Sections 1-3 (Motivation, Data Engineering, RLHF Pipeline)

---

## 📖 1. 核心动机与背景 (Introduction)

### 1.1 对齐问题 (The Alignment Problem)
*   **实然 (Is) vs. 应然 (Ought):**
    *   **GPT-3 (Pre-trained):** 训练目标是 **Next Token Prediction**。它模仿互联网上的海量数据，本质上是一个**“统计模拟器”**。它不仅模仿人类的知识，也模仿人类的偏见、谎言和废话。
    *   **InstructGPT (Aligned):** 用户的目标是获得一个**“有帮助、诚实、无害” (HHH)** 的助手。
*   **错位 (Misalignment):** 仅仅增加模型参数（Scale）并不能让模型更听话。相反，大模型可能更擅长编造令人信服的谎言（Hallucination）或生成有毒内容。
*   **目标:** 使用 **RLHF (Reinforcement Learning from Human Feedback)** 技术，将人类的意图（Intent）注入模型，使其输出分布从“互联网平均水平”收敛到“人类期望水平”。

---

## 💾 2. 数据工程：冷启动与三位一体 (Dataset Construction)

这是本论文最核心的工程贡献之一，解决了“先有鸡还是先有蛋”的**冷启动 (Cold Start)** 问题。

### 2.1 提示词来源 (Prompt Sources)
为了训练 InstructGPT，OpenAI 需要大量的 Prompt。来源分为两个阶段：
1.  **标注员手写 (Labeler-written):**
    *   **原因:** 在项目初期，API 用户主要把 GPT-3 当作“续写工具”，缺乏指令类数据。必须通过人工撰写来**自举 (Bootstrap)**。
    *   **三种类型:**
        *   **Plain (朴素指令):** 任意任务（如“列出 5 个诺贝尔奖得主”）。**逻辑目的:** 增加任务多样性。
        *   **Few-shot (少样本):** 指令 + 多个 Input/Output 示例。**逻辑目的:** 保持模型的 In-context Learning 能力，连接旧范式。
        *   **User-based (用户模拟):** 根据 API Waitlist 申请理由（如“想做食谱生成器”）撰写的 Prompt。**逻辑目的:** **Grounding (落地)**，确保解决真实世界需求。
2.  **API 用户数据 (Customer API):**
    *   随着 InstructGPT 的早期版本部署，收集真实用户的输入。
    *   **筛选:** 去重、去 PII（敏感信息）、按 User ID 截断（防止长尾用户主导）。

### 2.2 标注员团队 (The Labelers)
*   **筛选:** 通过 Upwork/ScaleAI 聘请了 **40 名** 承包商。
*   **标准:** 必须通过筛选测试（英语能力、逻辑推理、对 HHH 原则的理解）。
*   **一致性:** 标注员之间的打分一致性约为 73%。
*   **局限:** 主要是受过高等教育的西方人。这意味着模型学习的是**特定群体的价值观**，而非全人类的普遍真理。

### 2.3 三大核心数据集 (The Three Datasets)
这是理解 RLHF 流程的物理基础。请注意这三个数据集的**来源**和**格式**的区别：

| 数据集名称 | 样本量 | 来源构成 | 数据格式 (Input $\to$ Label) | 对应步骤 | 成本逻辑 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SFT Dataset** | **13k** | API + **大量手写** | `Prompt` $\to$ `Demonstration` (人工手写答案) | Step 1 | **极高** (创作难) |
| **RM Dataset** | **33k** | API + 少量手写 | `Prompt` $\to$ `Ranking` (对 $K$ 个生成结果排序) | Step 2 | **中等** (比较易) |
| **PPO Dataset** | **31k** | **仅 API** | `Prompt` only (无标签) | Step 3 | **极低** (无需人标) |

---

## 🔬 3. 方法论：RLHF 三步走详解 (Methodology)

Figure 2 是整篇论文的灵魂。它描述了一个从**模仿**到**评价**再到**进化**的逻辑闭环。

### 🟢 Step 1: 监督微调 (Supervised Fine-Tuning, SFT)
> **"Collect demonstration data, and train a supervised policy."**

*   **逻辑本质:** **行为克隆 (Behavior Cloning)**。解决“冷启动”问题，让模型先学会“像人一样说话”。
*   **基座:** GPT-3 (175B)。
*   **训练:** 标准的 Causal Language Modeling (CLM) 损失。
*   **关键细节:**
    *   只训练 **1 Epoch**。
    *   **原因:** SFT 数据集太小 (13k) 而模型太大。多练会导致严重的**过拟合**，虽然 SFT Loss 降低，但人类评分并没有提高。
*   **局限 (The Expertise Bottleneck):**
    *   模型的上限被标注员的写作能力锁死。标注员写不出高质量的代码，模型也学不会。

### 🔵 Step 2: 奖励模型 (Reward Modeling, RM)
> **"Collect comparison data, and train a reward model."**

*   **逻辑本质:** **价值观植入 (Value Implantation)**。利用“验证比生成容易”的不对称性，突破标注员的能力瓶颈。
*   **模型架构:**
    *   **6B 参数** (基于 SFT 模型初始化，去掉 Unembedding 层，加一个 Scalar Head)。
    *   **输入:** `(Prompt, Response)`
    *   **输出:** 一个标量分数 (Scalar Score)。
*   **为什么是 6B?**
    *   在 Step 3 中，RM 需要对每一个生成的 Token 序列打分。如果 RM 也是 175B，计算成本将不可接受。实验证明 6B 的判别能力已经足够。
*   **数据构造 ($K$ Trick):**
    *   让模型对一个 Prompt 生成 $K$ 个回答 ($K=4 \sim 9$)。
    *   标注员进行全排序 (Ranking)。
    *   一次标注产生 $\binom{K}{2}$ 个成对数据 (Pairs)。这极大地提高了数据效率。
*   **损失函数 (Pairwise Ranking Loss):**
    $$ \text{loss}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l))) \right] $$
    *   **逻辑:** 这是一个**序数效用 (Ordinal Utility)** 函数。只要 $y_w$ (胜者) 的分数比 $y_l$ (败者) 高，Loss 就低。不追求绝对分数的准确性。

### 🟣 Step 3: 强化学习 (Reinforcement Learning, PPO)
> **"Optimize a policy against the reward model using PPO."**

*   **逻辑本质:** **受限进化 (Constrained Optimization)**。在保持“人话”的前提下，追求“高分”。
*   **环境:** Bandit Environment。
*   **目标函数详解:**
    $$ \text{maximize } \mathbb{E} \left[ \underbrace{r_\theta(x, y)}_{\text{动力}} - \underbrace{\beta \log \left( \frac{\pi_{\phi}^{RL}(y|x)}{\pi^{SFT}(y|x)} \right)}_{\text{阻力 (KL Penalty)}} + \underbrace{\gamma \mathbb{E}_{x \sim D_{pretrain}} [\log(\pi_{\phi}^{RL}(x))]}_{\text{修正 (PPO-ptx)}} \right] $$

    1.  **Reward ($r_\theta$):** 鼓励模型生成 RM 认为好的回答。
    2.  **KL Penalty ($-\beta \log ...$):**
        *   计算 RL 模型与 SFT 模型输出概率的 **KL 散度**。
        *   **作用:** 防止 **Reward Hacking**（模型钻空子，输出乱码骗分）。它像一根绳子，把 RL 模型拴在 SFT 模型（语言通顺的模型）附近。
    3.  **PPO-ptx ($+\gamma ...$):**
        *   **Pretraining Mix:** 在 RL 更新时，混入原始的预训练数据（如 Wikipedia, Books）。
        *   **作用:** 缓解 **Alignment Tax (对齐税)**。防止模型为了讨好人类指令，而遗忘了物理、历史等基础知识。

---

## 🧠 4. 逻辑学视角的阶段性总结

读完前三章，我们构建了 InstructGPT 的**本体论**：

1.  **智能的来源:**
    *   **能力 (Capability):** 来自预训练 (GPT-3)。
    *   **形式 (Form):** 来自 SFT (Step 1)。
    *   **偏好 (Preference):** 来自 RM (Step 2)。
    *   **策略 (Policy):** 来自 PPO (Step 3)。

2.  **真理的相对性:**
    *   InstructGPT 没有追求绝对真理，它追求的是 **"OpenAI 雇佣的 40 个标注员眼中的好"**。这是一个**主观唯心**的系统，而非客观唯物的系统。

3.  **验证与生成的二元对立:**
    *   Step 1 依赖人类的**生成能力** (Demonstration)，这是昂贵且有限的。
    *   Step 2 依赖人类的**验证能力** (Comparison)，这是廉价且上限更高的。
    *   InstructGPT 的成功，本质上是**验证驱动生成 (Verification-driven Generation)** 的胜利。