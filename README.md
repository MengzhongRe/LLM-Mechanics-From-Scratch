# 🧠 Neuro-Symbolic AI & LLM Mechanics: From Papers to PyTorch

[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()
[![Focus](https://img.shields.io/badge/Focus-LLM_%26_Reasoning-blue?style=flat-square)]()
[![Author](https://img.shields.io/badge/Author-SYSU_Logic_Master-purple?style=flat-square)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch)]()

> "What I cannot create, I do not understand." — Richard Feynman
> "Reading papers is not just about gaining knowledge, but about deconstructing the logic behind intelligence."

## 📖 Introduction (项目简介)

This repository serves as my personal knowledge base and engineering portfolio, documenting my 6-month hardcore journey from **Formal Logic** to **Deep Learning Engineering**.

As a Logic Master (SYSU), I am particularly interested in the intersection of **Neuro-Symbolic AI**:
*   **Classic Architectures:** Deconstructing the vector space semantics and tearing down SOTA models.
*   **Reasoning Abilities:** How strict logical deduction emerges from probabilistic next-token prediction.
*   **Alignment & Safety:** Mapping human values (Ethics/Law) into model constraints.

## 📂 Repository Structure (目录结构)

- `01_Paper_Notes/`: In-depth Markdown notes breaking down the math, architecture, and logic of SOTA AI papers.
- `02_Handwritten_Operators/`: Pure PyTorch implementations of core operators from scratch (No `transformers` library, just math and tensors).
- `thoughts/`: Essays on "Logic vs. Neural Networks".

---

## 🚀 The Matrix: Paper to Code (核心算子纯手工复现)

*Show, Don't Tell.* Translating mathematical formulas from SOTA papers into executable PyTorch tensors.

| Domain | Paper / Topic | Core Concept | PyTorch Implementation | Note Link |
| :--- | :--- | :--- | :--- | :--- |
| **Foundation** | [Attention is all you need] | Scaled Dot-Product, MHA | [`multi_head_attention.py`](./02_Handwritten_Operators/multi_head_attention.py) | [Note](./01_Paper_Notes/Foundation/2017_Attention.md) |
| **Foundation** | Numerical Stability | Safe Softmax |[`safe_softmax.py`](./02_Handwritten_Operators/safe_softmax.py) | N/A |
| **Architecture** |[Flash Attention]| online_softmax| [`online_softmax.py`](./02_Handwritten_Operators/online_softmax.py) | [Note](./01_Paper_Notes/Modern_Architecture/2022_Flash_Attention.md) |


*(Status: Continuously updating on Fridays...)*

---

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
| 2026-02-12 | **Training Verifiers to Solve Math Word Problems** |[PDF](https://arxiv.org/abs/2110.14168) | `Verifier`, `Reward_Modelling` | ✅ Done |

### Phase 5: Neuro-Symbolic, Code & Math (The "Logic")
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-02-13 | **PAL: Program-aided Language Models** | [PDF](https://arxiv.org/abs/2211.10435) | `Code`, `Symbolic` | ✅ Done |
| 2026-02-14 | **Let's Verify Step by Step (Process Reward Models)** | [PDF](https://arxiv.org/abs/2305.20050) | `PRM`, `Math`, `SOTA` | ✅ Done |
| 2026-02-15 | **LeanDojo: Theorem Proving with Retrieval-Augmented LLMs** |[PDF](https://arxiv.org/abs/2306.15626) | `Theorem_Proving`, `Formal` | 📅 Planned |
| 2026-03-02 | **AlphaGeometry: Solving Olympiad Geometry** | [Nature](https://www.nature.com/articles/s41586-023-06747-5) | `Neuro-Symbolic`, `IMO` | ✅ Done |
| 2026-03-03 | **Maieutic Prompting: Logically Consistent Reasoning** | [PDF](https://arxiv.org/abs/2205.11822) | `Maieutic`, `Consistency` | ✅ Done |
| 2026-03-04 | **Least-to-Most Prompting Enables Complex Reasoning** | [PDF](https://arxiv.org/abs/2205.10625) | `Decomposition`, `Symbolic` | ✅ Done |

### Phase 6: Modern Architectures (The "SOTA")
| Date | Paper Title | Links | Tags | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-03-04 | **LLaMA: Open and Efficient Foundation Language Models** | [PDF](https://arxiv.org/abs/2302.13971) | `LLaMa`,`RoPE`, `SwiGLU` | ✅ Done |
| 2026-03-05 | **Mixtral of Experts** | [PDF](https://arxiv.org/abs/2401.04088) | `MoE`, `Sparse_Activation` | ✅ Done |
| 2026-03-06 | **Lost in the Middle: How Language Models Use Long Contexts** | [PDF](https://arxiv.org/abs/2307.03172) | `Long_Context` | ✅ Done |
| 2026-03-07 | **FlashAttention: Fast and Memory-Efficient Exact Attention** | [PDF](https://arxiv.org/abs/2205.14135) | `IO-Aware`, `Optimization` | ✅ Done |
| 2026-03-08 | **GQA: Training Generalized Multi-Query Transformer Models** |[PDF](https://arxiv.org/abs/2305.13245) | `KV_Cache`, `Inference` | 📅 Planned |
| 2026-03-09 | **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** | [PDF](https://arxiv.org/abs/2312.00752) | `SSM`, `Beyond_Transformer`| 📅 Planned |
| 2026-03-10 | **Visual Instruction Tuning (LLaVA)** | [PDF](https://arxiv.org/abs/2304.08485) | `Multimodal`, `VLM` | 📅 Planned |

---

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