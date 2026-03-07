# [Paper Read] ReAct: Synergizing Reasoning and Acting in Language Models
- **Authors**: Shunyu Yao (Princeton), Jeffrey Zhao (Google Brain), et al.
- **Year**: 2023 (ICLR)
- **Link**: [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- **Tags**: #Agent #Tool-Use #Reasoning #Interleaved-Generation #Decision-Making

---

## 1. 核心动机 (Motivation)

LLM 存在两类割裂的模式：
1.  **推理 (Reasoning):** 如 CoT。擅长逻辑推导，但不仅限于内部参数知识，容易产生**事实幻觉 (Hallucination)**，且无法获取最新信息。
2.  **行动 (Acting):** 如传统 RL/WebGPT。擅长与环境交互，但缺乏**意向性 (Intentionality)** 和**规划 (Planning)**，容易变成无头苍蝇。

**ReAct (Reason + Act)** 提出将“思考”与“行动”交织（Interleave）在一起，形成 **"Thought $\rightarrow$ Action $\rightarrow$ Observation"** 的闭环。

## 2. 核心方法 (The Method)

### 2.1 逻辑形式化
将问题解决过程建模为序列 $\tau = (c, th_1, a_1, o_1, th_2, a_2, o_2, \dots)$：
- **$th_t$ (Thought):** 模型的内省思考。用于分解目标、跟踪状态、纠正错误。不影响外部环境。
- **$a_t$ (Action):** 模型输出的指令（如 `Search[entity]`）。用于与外部交互。
- **$o_t$ (Observation):** 环境返回的真实反馈（如 Wikipedia 的搜索结果）。

### 2.2 控制流 (Control Flow)
ReAct 并非纯端到端生成，而是依赖 **Stop Sequence** 的半自动控制：
1.  模型生成 Thought 和 Action。
2.  遇到 `Action:` 后暂停。
3.  外部环境（Python脚本）执行动作，并将结果以 `Observation:` 的形式拼接回 Context。
4.  模型读取 Observation，更新状态，继续生成下一个 Thought。

## 3. 实验发现 (Key Experiments)

### 3.1 知识密集型任务 (HotPotQA, FEVER)
- **ReAct vs CoT:** ReAct 解决了 CoT 的幻觉问题，通过引入外部证据（Grounding），在事实验证任务上显著优于 CoT。
- **ReAct vs Act-Only:** ReAct 通过 Thought 提供了**目标导向**，避免了漫无目的的搜索。
- **组合策略:** **CoT-SC $\rightarrow$ ReAct**。对于模型确定的知识直接回答（省钱），不确定的知识调用 ReAct（求真）。

### 3.2 决策型任务 (ALFWorld, WebShop)
- **状态跟踪:** 在文字冒险游戏中，Thought 充当了**显式短期记忆**，记录了“我现在在哪”、“我手里有什么”。
- **动态调整:** 当 Action 失败（如“门锁了”）时，Thought 能进行归因分析并修正计划（“我需要找钥匙”），而非陷入死循环。
- **数据效率:** ReAct (Few-shot) 的效果能够匹敌甚至超越需要数万次训练的强化学习 (RL) 模型。

### 3.3 微调的反转 (The Finetuning Reversal)
- **Prompting:** 小模型（8B）很难通过 Few-shot 学会 ReAct 的复杂格式。
- **Fine-tuning:** 一旦使用 ReAct 轨迹数据对小模型进行微调，效果**逆袭**，超越了单纯 Prompting 的大模型（62B/540B）。
    - *逻辑学启示:* 微调 CoT 只是在灌输**事实 (Fact)**，而微调 ReAct 是在灌输**方法论 (Methodology)**。

---

## 4. 逻辑学视角 (Logical Analysis)

**1. 动态认知逻辑 (Dynamic Epistemic Logic)**
ReAct 打破了 CoT 的静态推理封闭环。每一次 `Observation` 都是一次**信念更新 (Belief Update)**。模型不再是闭门造车的哲学家，而是通过实验（Action）来证伪或证实假设的科学家。

**2. 意向性与因果 (Intentionality)**
`Thought` 模块赋予了模型**意向性**。每一个 `Action` 不再是随机的概率分布，而是由前序 `Thought`（意图）逻辑蕴含的必然结果。

---

## 5. 工程启示 (Engineering Takeaway)

- **Agent 的雏形:** ReAct 是现代 Agent 框架（如 LangChain）的理论鼻祖。
- **算力与数据的权衡:** 对于垂直领域（如法律/金融），与其追求超大模型，不如构建高质量的 **ReAct SFT 数据** 去微调一个中等模型 (7B/14B)。这是 5070Ti 等消费级显卡落地的最佳路径。