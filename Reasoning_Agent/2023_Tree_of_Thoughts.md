# [Paper Read] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

- **Authors**: Shunyu Yao (Princeton), Dian Yu (Google DeepMind), et al.
- **Year**: 2023 (NeurIPS)
- **Link**: [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- **Tags**: #Reasoning #Search-Algorithm #System2 #BFS #DFS

---

## 1. 动机与背景 (Motivation)

- **System 1 vs System 2:**
    - 现有的 LLM（包括 CoT）类似于人类的 **System 1（快思考）**：直觉的、联想的、线性的、不可逆的。
    - 解决复杂问题（如数学证明、规划、创作）需要 **System 2（慢思考）**：逻辑的、规划的、可回溯的。
- **核心局限:** Token 级别的自回归生成（Autoregressive Generation）一旦在某一步做出了错误选择，很难自我纠正。

**本文核心贡献：** 将 LLM 的推理过程形式化为 **树状搜索 (Tree Search)**，赋予模型探索 (Exploration) 和前瞻 (Lookahead) 的能力。

---

## 2. 框架定义 (The ToT Framework)

ToT 将推理过程建模为在**状态空间 (State Space)** 中的搜索。一个完整的 ToT 系统包含四个模块：

### 2.1 思考分解 (Thought Decomposition)
为了适应 LLM 的生成特性，必须将复杂问题 $x$ 拆解为若干个中间思考步骤 $z_t$。
- **例如 (Game of 24):** 不直接生成公式，而是分三步，每步描述一个中间运算（"Use 4 and 9 to make 13"）。

### 2.2 思考生成器 (Thought Generator) - $G(p_\theta, s, k)$
给定当前状态 $s$，如何生成 $k$ 个候选的下一步 $z$？
- **策略 A (Sample):** 从 CoT 提示中独立采样 $k$ 次。适用于思维空间广阔的任务（如创意写作）。
- **策略 B (Propose):** 使用 "Propose Prompt" 让模型一次性列出 $k$ 个可能的下一步。适用于解空间受限的任务（如 24点、数独）。

### 2.3 状态评估器 (State Evaluator) - $V(p_\theta, S)$ 🌟 *Core*
这是 ToT 区别于 CoT 的关键。引入**启发式评估 (Heuristic Evaluation)** 来判断状态的好坏。
- **策略 A (Value):** 对状态进行打分或分类（Sure/Likely/Impossible）。
    - *Prompt:* "通过剩下的数字 {10, 13, 13} 凑出 24 有可能吗？"
    - 适用于有明确目标导向的任务（硬逻辑）。
- **策略 B (Vote):** 通过投票比较不同状态的优劣。
    - *Prompt:* "对比以下两个续写段落，哪个更有逻辑？"
    - 适用于难以量化但易于比较的任务（软逻辑）。

### 2.4 搜索算法 (Search Algorithm)
- **BFS (广度优先):** 每一步保留评估分最高的 $b$ 个状态。用于 24点、写作。
- **DFS (深度优先):** 优先探索当前最有希望的路径，若评估为 "Impossible" 则回溯 (Backtrack)。用于填字游戏。

---

## 3. 实验分析 (Experiments)

### 3.1 Game of 24 (算术推理)
- **设置:** 1362 个高难度题目。BFS ($b=5$)。
- **结果:**
    - **IO (Direct):** 7.3%
    - **CoT:** 4.0% (线性思维容易在第一步算错后无法回头)
    - **ToT:** **74.0%**
- **结论:** 对于需要多步运算且容错率低的任务，搜索+剪枝是必须的。

### 3.2 Creative Writing (创意写作)
- **设置:** 输入4个句子，要求写一段话，结尾必须依次是这4句。
- **评估:** 使用 GPT-4 进行 Zero-shot 评分 (1-10分) 和成对比较。
- **结果:** ToT 生成的文本在连贯性 (Coherency) 上显著优于 CoT。
- **逻辑:** "先规划 (Plan) 后写作" 比 "边写边想" 更符合长文本生成的逻辑结构。

### 3.3 Mini Crosswords (填字游戏)
- **设置:** $5\times5$ 的填字游戏。DFS 算法。
- **核心机制:** **Pruning (剪枝)**。如果填入单词后发现后续位置无词可填，立即回溯。
- **结果:** 字级准确率 (Letter Accuracy) 从 53% (CoT) 提升至 78% (ToT)。

---

## 4. 逻辑学视角 (Logical Analysis)

**1. 模态逻辑与可能世界 (Possible Worlds):**
ToT 本质上是在构建一个 Kripke 语义模型。当前状态 $s$ 是“现实世界”，候选状态 $z$ 是通过推理关系 $R$ 可达的“可能世界”。Evaluator 的作用是评估这些可能世界中蕴含真理（Goal）的概率。

**2. 反事实推理 (Counterfactual):**
DFS 中的回溯机制体现了反事实思维：“如果我刚才不填这个词，而是填那个词，结果会不会更好？” CoT 缺乏这种反思能力。

**3. 外部控制论 (External Control):**
ToT 证明了通过**外部符号系统 (Python Search Code)** 来约束 **统计连接主义模型 (LLM)**，可以实现更高级的认知功能。这是 Neuro-Symbolic AI 的一种初级形态。

---

## 5. 工程伪代码 (Pseudocode)

```python
# BFS Implementation Sketch
def bfs_solve(prompt, model, b=5):
    current_states = [prompt]
    for step in range(3):
        # 1. Generate: Propose next steps
        candidates = [model.propose(s) for s in current_states]
        
        # 2. Evaluate: Check feasibility
        scores = [model.evaluate(c) for c in candidates]
        
        # 3. Select: Prune bad paths
        current_states = select_top_k(candidates, scores, k=b)
    
    return best(current_states)
```