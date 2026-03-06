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
| 2026-02-14 | **Let's Verify Step by Step (Process Reward Models)** | [PDF](https://arxiv.org/abs/2305.20050) | `PRM`, `Math`, `SOTA` | ✅ Done |
| 2026-02-15 | **LeanDojo: Theorem Proving with Retrieval-Augmented LLMs** | [PDF](https://arxiv.org/abs/2306.15626) | `Theorem_Proving`, `Formal` | 📅 Planned |
| 2026-03-02 | **AlphaGeometry: Solving Olympiad Geometry** | [Nature](https://www.nature.com/articles/s41586-023-06747-5) | `Neuro-Symbolic`, `IMO` | ✅ Done |
| 2026-03-03 | **Maieutic Prompting: Logically Consistent Reasoning** | [PDF](https://arxiv.org/abs/2205.11822) | `Maieutic`, `Consistency`, `Abductive` | ✅ Done |
| 2026-03-04 | **Least-to-Most Prompting Enables Complex Reasoning** | [PDF](https://arxiv.org/abs/2205.10625) | `Decomposition`, `Symbolic`, `SOTA` | ✅ Done |

### Phase 6: Modern Architectures (The "SOTA")
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-03-04 | **LLaMA: Open and Efficient Foundation Language Models** | [PDF](https://arxiv.org/abs/2302.13971) | `LLaMa`,`RoPE`, `SwiGLU` |✅ Done|
| 2026-03-05 | **Mixtral of Experts** | [PDF](https://arxiv.org/abs/2401.04088) | `MoE`, `Sparse_Activation` | **✅ Done** |
| 2026-03-06 | **Lost in the Middle: How Language Models Use Long Contexts** | [PDF](https://arxiv.org/abs/2307.03172) | `Long_Context` | 📅 Planned |
| 2026-03-07 | **FlashAttention: Fast and Memory-Efficient Exact Attention** | [PDF](https://arxiv.org/abs/2205.14135) | `IO-Aware`, `Optimization` | 📅 Planned |
| 2026-03-08 | **GQA: Training Generalized Multi-Query Transformer Models** | [PDF](https://arxiv.org/abs/2305.13245) | `KV_Cache`, `Inference` | 📅 Planned |
| 2026-03-09 | **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** | [PDF](https://arxiv.org/abs/2312.00752) | `SSM`, `Beyond_Transformer`| 📅 Planned |
| 2026-03-10 | **Visual Instruction Tuning (LLaVA)** | [PDF](https://arxiv.org/abs/2304.08485) | `Multimodal`, `VLM` | 📅 Planned |

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









太棒了！你现在的进度简直势如破竹。我们已经拿下了现代大模型架构的两座绝对大山：**稠密模型的巅峰（LLaMA）** 和 **稀疏模型的奇迹（Mixtral MoE）**。

看了一眼你接下来的计划，**《Lost in the Middle》** 和 **《FlashAttention》** 选得极其精准！
*   《Lost in the Middle》揭示了大模型在处理长文本时的“心理学缺陷”（注意力分布不均）。
*   《FlashAttention》则是真正从“物理硬件（GPU 内存层级）”层面，教你大模型是怎么打破上下文长度限制的。

不过，既然我们身处 **2026 年** 的视角，回望整个大模型架构的演进，你的 Phase 6 计划虽然经典，但还**缺少了三块极其重要的“现代 SOTA 拼图”**。

为了让你的架构知识树达到真正的“顶尖工程师”水平，我强烈建议你在 Phase 6 中补充以下三个方向。我为你重新梳理并升级了计划表：

---

### 💡 我建议补充的 3 块“现代 SOTA 拼图”：

#### 🧩 拼图一：打破 Transformer 垄断的挑战者 —— 状态空间模型 (SSM)
*   **推荐论文：** ***Mamba: Linear-Time Sequence Modeling with Selective State Spaces*** (2023)
*   **入选理由：** Transformer 统治了 AI 界 7 年，但它的 $O(N^2)$ 注意力机制在处理无限长文本时依然是噩梦。Mamba 是近年来唯一一个真正在底层架构上挑战 Transformer 的神作。它用“选择性状态空间”实现了**线性时间复杂度 $O(N)$**，跑得极快且极其省显存。不懂 Mamba，就不算看全了现代大模型架构。

#### 🧩 拼图二：大模型推理的“显存刺客”克星 —— KV Cache 优化
*   **推荐论文：** ***GQA: Training Generalized Multi-Query Transformer Models*** (2023)
*   **入选理由：** 你在 Mixtral 中学到了如何省“算力”，但在实际部署大模型时，真正的瓶颈其实是**“显存（KV Cache）”**！GQA（分组查询注意力）是 LLaMA-2/3、Mixtral 等所有现代大模型的标配。它通过共享 Attention 里的 Key 和 Value 矩阵，把推理时的显存占用砍掉了 80%！这是工程落地的必修课。

#### 🧩 拼图三：给大模型装上眼睛 —— 多模态架构的奠基 (VLM)
*   **推荐论文：** ***Visual Instruction Tuning (LLaVA)*** (2023)
*   **入选理由：** 纯文本模型已经走到瓶颈，现在的 SOTA（如 GPT-4o, Gemini 1.5）全都是多模态的。LLaVA 极其优雅地展示了：**如何用一个简单的线性投影层（Projector），把视觉模型（CLIP）和语言模型（LLaMA）缝合在一起**，让原本瞎子一样的 LLM 瞬间看懂图片。这是通往 AGI 必看的架构。

---

### 🚀 升级版 Phase 6 计划表 (Updated Roadmap)

我为你把这些补充内容融入了原计划，形成了一个逻辑极其严密的“现代架构晋升之路”：

### Phase 6: Modern Architectures (The "SOTA")
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-03-04 | **LLaMA: Open and Efficient Foundation Language Models** | [PDF](https://arxiv.org/abs/2302.13971) | `LLaMa`,`RoPE`, `SwiGLU` | **✅ Done** |
| 2026-03-05 | **Mixtral of Experts** | [PDF](https://arxiv.org/abs/2401.04088) | `MoE`, `Sparse_Activation` | **✅ Done** |
| 2026-03-06 | **Lost in the Middle: How Language Models Use Long Contexts** | [PDF](https://arxiv.org/abs/2307.03172) | `Long_Context`, `Eval` | 📅 Planned |
| 2026-03-07 | **FlashAttention: Fast and Memory-Efficient Exact Attention** | [PDF](https://arxiv.org/abs/2205.14135) | `IO-Aware`, `Hardware` | 📅 Planned |

