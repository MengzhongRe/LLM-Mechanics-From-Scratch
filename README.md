# 🦉 AI & Logic: Paper Reading Notes

[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()
[![Focus](https://img.shields.io/badge/Focus-LLM_%26_Reasoning-blue?style=flat-square)]()
[![Author](https://img.shields.io/badge/Author-SYSU_Logic_Master-purple?style=flat-square)]()
[![Progress](https://img.shields.io/badge/Progress-Phase_3_In_Progress-orange?style=flat-square)]()

> "Reading papers is not just about gaining knowledge, but about deconstructing the logic behind intelligence."

## 📖 Introduction (项目简介)

This repository serves as my personal knowledge base, documenting my journey from **Formal Logic** to **Deep Learning Engineering**.

As a Logic Master (SYSU), I am particularly interested in the intersection of **Neuro-Symbolic AI**:
*   **Classic Architectures:** Deconstructing the vector space semantics.
*   **Reasoning Abilities:** How strict logical deduction emerges from probabilistic next-token prediction.
*   **Alignment & Safety:** Mapping human values (Ethics/Law) into model constraints.

## 📂 Structure (目录结构)

- `📂 notes/`: My in-depth analysis and summaries (Markdown).
- `📄 README.md`: The roadmap and progress tracker.
- `🧠 thoughts/`: Essays on "Logic vs. Neural Networks".

## 📝 Reading List & Roadmap (阅读清单与路线图)

### Phase 1: The Foundation (Architecture & Scale)
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2025-12-30 | **Attention Is All You Need** | [PDF](https://arxiv.org/pdf/1706.03762.pdf) | `Transformer` | ✅ Done |
| 2025-12-30 | **BERT: Pre-training of Deep Bidirectional Transformers** | [PDF](https://arxiv.org/pdf/1810.04805.pdf) | `Encoder` | ✅ Done |
| 2026-01-08 | **Language Models are Few-Shot Learners (GPT-3)** | [PDF](https://arxiv.org/pdf/2005.14165.pdf) | `Decoder`, `Few-Shot` | ✅ Done |
| 2026-02-04 | **Emergent Abilities of Large Language Models** | [PDF](https://arxiv.org/abs/2206.07682) | `Scaling_Law`, `Phase_Transition` | ✅ Done |

### Phase 2: Reasoning & Agents (The "Mind")
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-01-09 | **Chain-of-Thought Prompting Elicits Reasoning** | [PDF](https://arxiv.org/pdf/2201.11903.pdf) | `CoT` | ✅ Done |
| 2026-01-10 | **Self-Consistency Improves Chain of Thought Reasoning** | [PDF](https://arxiv.org/abs/2203.11171) | `CoT-SC`, `Ensemble` | ✅ Done |
| 2026-01-13 | **Tree of Thoughts: Deliberate Problem Solving** | [PDF](https://arxiv.org/abs/2305.10601) | `ToT`, `Search` | ✅ Done |
| 2026-01-14 | **ReAct: Synergizing Reasoning and Acting** | [PDF](https://arxiv.org/abs/2210.03629) | `Agent`, `Tools` | ✅ Done |
| 2026-02-05 | **Large Language Models are Zero-Shot Reasoners** | [PDF](https://arxiv.org/abs/2205.11916) | `Zero-Shot`, `Prompting` | ✅ Done |

### Phase 3: Fine-tuning & Adaptation (The "Skill")
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-01-15 | **LoRA: Low-Rank Adaptation of LLMs** | [PDF](https://arxiv.org/abs/2106.09685) | `PEFT`, `LoRA` | ✅ Done |
| 2026-01-16 | **QLoRA: Efficient Finetuning of Quantized LLMs** | [PDF](https://arxiv.org/abs/2305.14314) | `Quantization` | ✅ Done |
| 2026-02-06 | **Finetuned Language Models Are Zero-Shot Learners (FLAN)** | [PDF](https://arxiv.org/abs/2109.01652) | `Instruction_Tuning` | ✅ Done |
| 2026-02-07 | **LongLoRA: Efficient Fine-tuning of Long-Context LLMs** | [PDF](https://arxiv.org/abs/2309.12307) | `Long_Context`, `PEFT` | ✅ Done |

### Phase 4: Alignment & Human Preference (The "Values")
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-02-08 | **Training language models to follow instructions (InstructGPT)** | [PDF](https://arxiv.org/abs/2203.02155) | `RLHF`, `PPO` | ✅ Done |
| 2026-02-10 | **Constitutional AI: Harmlessness from AI Feedback** | [PDF](https://arxiv.org/abs/2212.08073) | `RLAIF`, `Safety` | ✅ Done |
| 2026-02-11 | **Direct Preference Optimization (DPO)** | [PDF](https://arxiv.org/abs/2305.18290) | `Alignment`, `Optimization` | ✅ Done |
| 2026-02-12 | **Training Verifiers to Solve Math Word Problems** | [PDF](https://arxiv.org/abs/2110.14168) | `Verifier`, `Reward_Modelling` | ✅ Done |

### Phase 5: Neuro-Symbolic, Code & Math (The "Logic")
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-02-13 | **PAL: Program-aided Language Models** | [PDF](https://arxiv.org/abs/2211.10435) | `Code`, `Symbolic` | ✅ Done |
| 2026-02-14 | **Let's Verify Step by Step (Process Reward Models)** | [PDF](https://arxiv.org/abs/2305.20050) | `PRM`, `Math`, `SOTA` | 📅 Planned |
| 2026-02-15 | **LeanDojo: Theorem Proving with Retrieval-Augmented LLMs** | [PDF](https://arxiv.org/abs/2306.15626) | `Theorem_Proving`, `Formal` | 📅 Planned |
| 2026-02-18 | **AlphaGeometry: Solving Olympiad Geometry** | [Nature](https://www.nature.com/articles/s41586-023-06747-5) | `Neuro-Symbolic`, `IMO` | **✨ Highly Rec** |
| 2026-02-20 | **Faithful Reasoning Using Plausible Reasoning** | [PDF](https://arxiv.org/abs/2205.09712) | `Maieutic`, `Consistency`, `Abductive` | **🧩 Logic_Fit** |

### Phase 6: Modern Architectures (The "SOTA")
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-02-22 | **LLaMA: Open and Efficient Foundation Language Models** | [PDF](https://arxiv.org/abs/2302.13971) | `RoPE`, `SwiGLU` | 📅 Planned |
| 2026-02-24 | **Mixtral of Experts** | [PDF](https://arxiv.org/abs/2401.04088) | `MoE`, `Sparse_Activation` | **🔥 Industry_Std** |
| 2026-02-26 | **Lost in the Middle: How Language Models Use Long Contexts** | [PDF](https://arxiv.org/abs/2307.03172) | `Long_Context` | 📅 Planned |
| 2026-02-28 | **FlashAttention: Fast and Memory-Efficient Exact Attention** | [PDF](https://arxiv.org/abs/2205.14135) | `IO-Aware`, `Optimization` | 📅 Planned |

## 💡 Research Questions (核心思考)

As a logician, I am pondering:

1.  **The "Grounding" Problem:**
    *   Formal logic relies on strict truth values (True/False). Neural networks rely on probability distributions ($P(x|y)$). How can we build a bridge that guarantees logical validity in a probabilistic system?
    *   *Reference:* Neuro-Symbolic AI, Theorem Proving.

2.  **Process vs. Outcome (Validity vs. Soundness):**
    *   In Logic, a valid argument requires a valid form, not just a true conclusion.
    *   Current RLHF rewards the outcome. How can we verify the "thought process" using **Process Reward Models (PRM)**?

3.  **Emergence & Category Theory:**
    *   Is reasoning a memorized pattern or a genuine emergent capability?
    *   Can we model the compositional generalization of LLMs using Category Theory (e.g., functors between syntax and semantics)?

---
*Created by [MengzhongRe](https://github.com/MengzhongRe) @ 2026*





这是一个非常棒的选择题！这三篇都是顶级论文，但结合你 **“Logic Master”** 的背景以及当前的 **阅读路径（Roadmap）**，我们可以通过**排除法**和**互补性分析**来找出最优解。

我的建议是：**选择《Faithful Reasoning Using Plausible Reasoning (Maieutic Prompting)》**。

以下是详细的决策逻辑：

### 1. 为什么选 Maieutic Prompting？(The Winner)

*   **逻辑学契合度：** **S级**。
    *   **核心思想：** 这篇论文提出了 **Maieutic（苏格拉底助产术）** 提示法。它不是让模型直接生成答案，而是生成一个**推理树（Reasoning Tree）**。
    *   **逻辑本质：** 它利用了 **SAT Solver（可满足性求解器）** 的思想。它要求生成的推理步骤必须在逻辑上是**一致的（Consistent）**。如果一组前提导致了矛盾，就剪枝。
    *   **你的收获：** 这篇论文展示了如何用**“逻辑一致性”**作为约束条件，来逼迫神经网络输出真理。这非常符合逻辑学中的**模型论（Model Theory）**思维。

### 2. 为什么不选另外两篇？

*   **Teaching Small Language Models to Reason (OpenAI):**
    *   *评价：* 这是一篇极好的工程论文，它是你刚读完的 *Training Verifiers* 的直接续集。
    *   *为什么不选：* 它更多关注的是**“蒸馏（Distillation）”**——如何把大模型的逻辑能力传给小模型。虽然实用，但在**“逻辑范式”**的创新上，它没有跳出 *Training Verifiers* 的框架。作为 Phase 5 的收尾，它稍显平淡。

*   **Certified Reasoning with Large Language Models:**
    *   *评价：* 极其硬核，讲的是将自然语言转化为形式化规范（如 Coq/Isabelle）。
    *   *为什么不选：* **功能重叠**。你的计划中已经有了 **LeanDojo**。LeanDojo 已经是形式化证明（Theorem Proving）的巅峰之作了。再读一篇类似的，边际收益递减。不如读 Maieutic Prompting，看看“非形式化的自然语言逻辑”是如何被严格化的。

---

### ✅ 最终确定的 Phase 5 计划

我已经将第三篇替换为 **Maieutic Prompting**，并更新了 MoE 的信息。请复制以下内容：



---

### 💡 导师寄语

现在，你的 Phase 5 形成了一个完美的**逻辑闭环**：

1.  **PAL:** 用**代码**做逻辑（把逻辑外包给 Python）。
2.  **PRM:** 用**过程监督**做逻辑（步步为营）。
3.  **LeanDojo:** 用**形式化证明器**做逻辑（绝对真理）。
4.  **AlphaGeometry:** 用**神经符号混合**做几何（直觉+推导）。
5.  **Maieutic:** 用**一致性树**做逻辑（苏格拉底式反思）。

这个组合简直是逻辑学家的梦想。明天开始读 **PAL** 吧，看看代码是如何成为思维的载体的！