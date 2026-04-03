# 🧠 LLM-Mechanics-From-Scratch

[![Status](https://img.shields.io/badge/Status-Hardcore_Engineering-success?style=flat-square)]()
[![Focus](https://img.shields.io/badge/Focus-LLM_Mechanics_%26_Reasoning-blue?style=flat-square)]()
[![Author](https://img.shields.io/badge/Author-SYSU_Logic_Master-purple?style=flat-square)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch_%7C_Triton-EE4C2C?style=flat-square&logo=pytorch)]()

> "What I cannot create, I do not understand." — Richard Feynman
> "To ground the discrete symbols of Formal Logic into the continuous vector space of Neural Networks, one must first tear down the matrices."

## 📖 Introduction (项目简介)

Welcome to my personal engineering workshop. This repository documents my hardcore journey from **Formal Logic** to **Deep Learning Engineering / LLM Architecture**.

**[Update 2026.03]** After 6 months of rigorous literature review (now archived in `01_Paper_Notes`), this repository has officially transitioned into the **"Hardcore Engineering Phase"**. 

My ultimate goal as a Logician is to build a **Neuro-Symbolic Reasoning Engine**. To achieve this, I am spending 65 days writing core LLM operators from scratch (Test-Driven Development), focusing on memory bound limits, precision conversions, distributed scaling, and Reinforcement Learning (GRPO) for logical reasoning.

## 📂 Repository Structure (架构与文档哲学)

This repository strictly follows the **"3D Documentation Organization"** tailored for production-grade engineering:
1. **Micro (Code Level):** Rich Docstrings & inline comments explaining Tensor shape transformations, `contiguous` traps, and memory management.
2. **Meso (Math Level):** Jupyter Notebooks (`.ipynb`) used as scratchpads for complex math derivations (e.g., Complex numbers in RoPE).
3. **Macro (Module Level):** A `README.md` in each sub-directory summarizing architectural comparisons and engineering pitfalls.

```text
📦 LLM-Mechanics-From-Scratch
 ┣ 📂 02_Handwritten_Operators/   # [🔥 Active] Pure PyTorch/Triton implementations (No HuggingFace)
 ┃ ┣ 📂 Phase0_Tokenization/      # BPE Tokenizer
 ┃ ┣ 📂 Phase1_Backbone/          # Foundation (RoPE, SwiGLU, RMSNorm, MoE...)
 ┃ ┣ 📂 Phase2_Inference/         # Memory & Speed (GQA, MLA, PagedAttention, W8A8...)
 ┃ ┣ 📂 Phase3_Decoding/          # Search & Sampling (Speculative Decoding, MCTS...)
 ┃ ┗ 📂 Phase4_Alignment_Scaling/ # RL & Distributed (TP Mock, LoRA, DPO, GRPO...)
 ┣ 📂 03_Nano_Logic_Engine/       # [🚧 WIP] A 50M custom LLM trained with Rule-based GRPO
 ┣ 📂 01_Paper_Notes/             # [✅ Archived] Markdown notes of 25+ SOTA AI papers
 ┗ 📂 thoughts/                   # Essays on "Logic vs. Neural Networks"
```

---

## 🚀 The Matrix: "Code-First" Mechanics Roadmap (核心算子手撕路线)

> **执行准则 (Test-Driven Development)**: Write pure PyTorch/Triton code $\rightarrow$ Verify against HF Ground Truth via `torch.allclose(atol=1e-5)` $\rightarrow$ Optimize for memory (`in-place`, `bf16`).

### Phase 0: The Discrete Grounding (数据与标记化)
*目标：理解大模型的第一步，将离散的逻辑符号映射为连续空间的 Token ID。*

| 天数 | 核心主题与手撕代码 | 核心阅读论文与直达链接 (重点精读) | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 1-3** | **BPE Tokenizer**<br>[`bpe_tokenizer.py`]() | **NMT of Rare Words (BPE)**<br>🔗 [PDF](https://arxiv.org/pdf/1508.07909.pdf) <br>🎯 *精读 Sec 3.2 (Algorithm 1)* | 1. 实现字符级BPE算法主要逻辑（统计词频、合并词对等）。<br>2. 重点实现 `byte_to_unicode()` 映射函数，注意处理特殊 `<think>` 标签。 |

### Phase 1: 现代大模型骨架与底层直觉 (Modern Backbone & Triton)
*目标：严格按照前向传播的数据流，重构带有 MoE 的现代大模型架构。*

| 天数 | 核心主题与手撕代码 | 核心阅读论文与直达链接 (重点精读) | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 4-6** | **RMSNorm & Compilation**<br>[`rmsnorm_triton.py`]() | **RMSNorm**<br>🔗 [PDF](https://arxiv.org/pdf/1910.07467.pdf) <br>🎯 *精读 Sec 3 (公式 3/4)* | 1. 理解去掉均值 $\mu$ 的数学依据。<br>2. 观察 `torch.compile` 的图融合，手写 Triton Kernel 掌握 Shared Memory。 |
| **Day 7-10** | **RoPE & Decoupled RoPE**<br>[`rope_embedding.py`]() | **RoFormer**<br>🔗 [PDF](https://arxiv.org/pdf/2104.09864.pdf) <br>🎯 *精读 Sec 3.4 (公式 34)* | 1. 预计算 Cos/Sin 缓存，避免重复计算。<br>2. **[为MLA铺垫]** 解耦 RoPE：只对 Q/K 的部分维度旋转。<br>3. `bf16` 输入，`fp32` 旋转以保精度。 |
| **Day 11-13** | **MHA & Safe Softmax**<br>[`mha_forward.py`]() | **Attention Is All You Need**<br>🔗[PDF](https://arxiv.org/pdf/1706.03762.pdf) <br>🎯 *精读 Sec 3.2* | 1. 手写因果掩码 (Causal Mask) 的零拷贝广播机制。<br>2. `safe_softmax` 中使用 `x - max(x)` 技巧防止溢出。 |
| **Day 14-16** | **🔥 SwiGLU FFN**<br>*(大模型的记忆细胞)*<br>[`swiglu_ffn.py`]() | **GLU Variants**<br>🔗 [PDF](https://arxiv.org/pdf/2002.05202.pdf) <br>🎯 *精读 Sec 2 (公式 5/6)* | 1. 手撕“先升维(Up/Gate)再降维(Down)”架构，理解 Key-Value 知识存储机制。<br>2. **[官方实现对齐]** 手写两层Naive版本的FFN，再对齐LLaMA官方的w1.w3权重合并式、与w2特殊初始化的方法，对比两者在运算速度上的区别,并对比compile和手撕的运算区别 |
| **Day 17-19** | **🔥 MoE Router & Experts**<br>[`moe_layer.py`]() | **Mixtral 8x7B**<br>🔗[PDF](https://arxiv.org/pdf/2401.04088.pdf) <br>🎯 *精读 Sec 2.1* | 1. 将前几天写的 `SwiGLU` 实例化为 8 个 Expert。<br>2. 编写 Top-K Router 得到概率权重。<br>3. **[算子核心]** 手写 `scatter` 分发与 `gather` 组合逻辑。 | <br>4. 在forward函数中手写负载均衡损失aux_loss,使得MoE模型能够真正发挥作用 |<br>5. 手写向量化版本的MoELayer并编写测试用例测试其与for循环Naive版本之间的推理耗时区别并进行逻辑等价性校验 ｜<br>6.编写代码对比负载均衡系数分别在0.01、100.0时，分析导致Router和Expert层之间梯度变化有什么不同 |
| **Day 20** | **CE Loss & Output Head**<br>[`loss_head.py`]() | **Language Models (GPT-3)** | 用 `LogSumExp` 技巧手撕稳健的 CrossEntropyLoss，完成从隐状态到词表概率分布的最后映射。 |

### Phase 2: 推理加速与极致显存魔术 (Inference & Memory Magic)
*目标：攻克 2026 年大厂面试中占比极高（近乎 100% 必考）的显存管理、FlashAttention 思想与 MLA。*

| 天数 | 核心主题与手撕代码 | 核心阅读论文与直达链接 (重点精读) | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 21-23** | **Online Softmax Mock**<br>[`online_softmax.py`]() | **FlashAttention**<br>🔗 [PDF](https://arxiv.org/pdf/2205.14135.pdf) <br>🎯 *精读 Algorithm 1* | 不要求写 CUDA，但必须用 Python `for` 循环按 Block (Tiling) 模拟 $m_i, l_i$ 的更新过程，理解 IO-Aware。 |
| **Day 24-27** | **Stateful KV & GQA**<br>[`gqa_kv_cache.py`]() | **GQA**<br>🔗 [PDF](https://arxiv.org/pdf/2305.13245.pdf) <br>🎯 *精读 Sec 2* | 1. 严格区分 **Prefill** 与 **Decode** 的张量流转。<br>2. 用 `repeat_interleave` 将 K/V 广播匹配 Q 的头数。 |
| **Day 28-31** | **PagedAttention Mock**<br>[`paged_attention.py`]() | **PagedAttention (vLLM)**<br>🔗[PDF](https://arxiv.org/pdf/2309.06180.pdf) <br>🎯 *精读 Sec 3.1 & Fig 4* | 用 Python 模拟 Block Table：给定逻辑 Token 索引，通过 `torch.gather` 从不连续的物理显存池中拼装 K/V。 |
| **Day 32-35** | **🔥 MLA Attention**<br>[`mla_attention.py`]() | **DeepSeek-V2**<br>🔗 [PDF](https://arxiv.org/pdf/2405.04434.pdf) <br>🎯 *精读 Sec 2.1.2 (公式 16-23)* | **大厂必考！** 写出将 $c_t$ (Latent Vector) 投影重构并吸收 (Absorb) 权重的逻辑，验证极小的 KV Cache 占用。 |
| **Day 36-38** | **W8A8 Quantization**<br>[`naive_quant.py`]() | **SmoothQuant**<br>🔗[PDF](https://arxiv.org/pdf/2211.10438.pdf) <br>🎯 *精读 Sec 3* | 手撕非对称 (Asymmetric) 量化公式：计算 Scale 和 Zero-point，完成 `x_q = clamp(round(x/s + z))` 及反量化。 |

### Phase 3: 解码、推测与树搜索 (Decoding & Speculative Execution)
*目标：从概率分布到 Token，并掌握大厂最省机器算力的解码黑科技。*

| 天数 | 核心主题与手撕代码 | 核心阅读论文与直达链接 (重点精读) | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 39-41** | **Top-P / Top-K Sampler**<br>[`sampler.py`]() | **Neural Text Degeneration**<br>🔗[PDF](https://arxiv.org/pdf/1904.09751.pdf) <br>🎯 *精读 Sec 4* | 给定 Logits，手写截断（`topk`）与核采样（先 `sort` 再 `cumsum` 做 Mask），最后 `torch.multinomial` 采样。 |
| **Day 42-45** | **Beam Search & MCTS**<br>[`beam_mcts.py`]() | **Tree of Thoughts**<br>🔗 [PDF](https://arxiv.org/pdf/2305.10601.pdf) <br>🎯 *精读 Sec 3* | 1. 维护大小为 $B$ 的堆进行束搜索。<br>2. **[o1前置]** 基于 UCB (Upper Confidence Bound) 写一个启发式节点选择器。 |
| **Day 46-49** | **🔥 Speculative Decoding**<br>[`speculative.py`]() | **Speculative Decoding**<br>🔗[PDF](https://arxiv.org/pdf/2211.17192.pdf) <br>🎯 *精读 Algorithm 1* | 初始化小 Draft 和大 Target 模型。实现 Draft 生成 $K$ 个 token，Target 一次前向后进行 **Accept/Reject 并行拒绝采样**的概率调整逻辑。 |

### Phase 4: 后训练：强化学习与分布式并行 (Alignment & Scaling)
*目标：告别“模仿人类”，转向“逼迫模型进行严谨数学/逻辑推理”。*

| 天数 | 核心主题与手撕代码 | 核心阅读论文与直达链接 (重点精读) | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 50-52** | **Tensor Parallel (Mock)**<br>[`tp_linear.py`]() | **Megatron-LM**<br>🔗 [PDF](https://arxiv.org/pdf/1909.08053.pdf) <br>🎯 *精读 Sec 3 (Fig 3)* | 写出 `ColumnParallel` 和 `RowParallel` 的前向逻辑，并在正确的位置插入 `fwd_all_reduce` 占位符（模拟跨卡通信）。 |
| **Day 53-54** | **LoRA Core Forward**<br>[`lora_linear.py`]() | **LoRA**<br>🔗[PDF](https://arxiv.org/pdf/2106.09685.pdf) <br>🎯 *精读 Sec 4.1 (公式 1-2)* | 初始化 $A$ (Normal) 与 $B$ (Zero)，实现 `scaling = alpha / r` 的前向逻辑与 `merge_weights` 权重融合。 |
| **Day 55-57** | **DPO Loss**<br>[`dpo_loss.py`]() | **DPO**<br>🔗[PDF](https://arxiv.org/pdf/2305.18290.pdf) <br>🎯 *精读 Sec 4 (公式 7)* | 输入 4 个 Logps（赢/输 x 策略/参考模型），写出基于 $\beta$ 放缩的对数 Sigmoid 隐式偏好损失。 |
| **Day 58-60** | **🔥 GRPO Loss**<br>[`grpo_loss.py`]() | **DeepSeekMath** 🔗 [PDF](https://arxiv.org/pdf/2402.03300.pdf)<br>**DeepSeek-R1** 🔗 [PDF](https://arxiv.org/pdf/2501.12948.pdf) | **RL 推理核心！** 摒弃 Critic 网络：对同 Prompt 采样 $N$ 个回答，计算 Reward 的组内均值和标准差，归一化得到 Advantage (优势函数)。 |

### Phase 5: 终极交付 —— "Neuro-Symbolic Reasoning" 引擎
*目标：拼装前 60 天的算子，结合逻辑学背景，打造极具辨识度的 RL-for-Reasoning 闭环 Demo。*

| 天数 | 核心动作 (Action) | 交付物模块 | 面试讲解锚点与验收标准 (Deliverable) |
| :--- | :--- | :--- | :--- |
| **Day 61-62** | **模型拼装与 Pretrain** | `nano_logic_model.py` | 用手写的 RoPE/RMSNorm/MoE/MLA 拼装一个 **50M 的微型大模型**，跑通极简 Training Loop。 |
| **Day 63** | **合成逻辑数据集** | `logic_data_gen.py` | **发挥 Logic 硕士护城河：** 编写脚本自动生成形式逻辑推演题（三段论/肯定前件式等），强制插入 `<think>` 模板。 |
| **Day 64-65** | **Rule-based GRPO** | `train_grpo_logic.py` | 1. 写一个 **Python 逻辑解析器** 作为确定性 Reward：推演符合严格逻辑规则 +1.0，格式错 -1.0。<br>2. 运行 GRPO 训练，展示微型模型如何通过纯 RL 涌现出 "Aha Moment" 和自我纠错能力。 |

---

<details>
<summary><h2>📚 [Archived] The Reading Phase (6-Month Paper Roadmap)</h2></summary>

*I spent 6 months thoroughly deconstructing the math and architecture of the following papers. The detailed Markdown notes can be found in `01_Paper_Notes/`.*

| Date | Paper Title | Tags | Status |
| :--- | :--- | :--- | :--- |
| 2025-12-30 | **Attention Is All You Need** | `Transformer` | ✅ Done |
| 2025-12-30 | **BERT: Pre-training of Deep Bidirectional Transformers** | `Encoder` | ✅ Done |
| 2026-01-08 | **Language Models are Few-Shot Learners (GPT-3)** | `Decoder`, `Few-Shot` | ✅ Done |
| 2026-02-04 | **Emergent Abilities of Large Language Models** | `Scaling_Law` | ✅ Done |
| 2026-01-09 | **Chain-of-Thought Prompting Elicits Reasoning** | `CoT` | ✅ Done |
| 2026-01-10 | **Self-Consistency Improves Chain of Thought Reasoning** | `CoT-SC` | ✅ Done |
| 2026-01-13 | **Tree of Thoughts: Deliberate Problem Solving** | `ToT`, `Search` | ✅ Done |
| 2026-01-14 | **ReAct: Synergizing Reasoning and Acting** | `Agent`, `Tools` | ✅ Done |
| 2026-02-05 | **Large Language Models are Zero-Shot Reasoners** | `Zero-Shot` | ✅ Done |
| 2026-01-15 | **LoRA: Low-Rank Adaptation of LLMs** | `PEFT`, `LoRA` | ✅ Done |
| 2026-01-16 | **QLoRA: Efficient Finetuning of Quantized LLMs** | `Quantization` | ✅ Done |
| 2026-02-06 | **Finetuned Language Models Are Zero-Shot Learners (FLAN)** | `Instruction_Tuning` | ✅ Done |
| 2026-02-08 | **Training language models to follow instructions (InstructGPT)** | `RLHF`, `PPO` | ✅ Done |
| 2026-02-11 | **Direct Preference Optimization (DPO)** | `Alignment` | ✅ Done |
| 2026-02-14 | **Let's Verify Step by Step (Process Reward Models)** | `PRM`, `Math` | ✅ Done |
| 2026-03-02 | **AlphaGeometry: Solving Olympiad Geometry** | `Neuro-Symbolic` | ✅ Done |
| 2026-03-04 | **LLaMA: Open and Efficient Foundation Language Models** | `LLaMa`,`RoPE` | ✅ Done |
| 2026-03-06 | **Lost in the Middle: How Language Models Use Long Contexts** | `Long_Context` | ✅ Done |
| 2026-03-10 | **FlashAttention: Fast and Memory-Efficient Exact Attention** | `IO-Aware` | ✅ Done |
| 2026-03-12 | **GQA: Training Generalized Multi-Query Transformer Models** | `KV_Cache` | ✅ Done |
| 2026-03-16 | **Visual Instruction Tuning (LLaVA)** | `Multimodal`, `VLM` | ✅ Done |

</details>

---

## 💡 Research Questions (核心思考)

As a logician, I am pondering:
1.  **The "Grounding" Problem:** Formal logic relies on strict truth values (True/False). Neural networks rely on probability distributions ($P(x|y)$). How can we build a bridge that guarantees logical validity in a probabilistic system?
2.  **Process vs. Outcome (Validity vs. Soundness):** In Logic, a valid argument requires a valid form, not just a true conclusion. Current RLHF rewards the outcome. How can we verify the "thought process" using **Rule-based Process Reward Models (PRM)**?

---
*Created by[MengzhongRe](https://github.com/MengzhongRe) @ 2026*
