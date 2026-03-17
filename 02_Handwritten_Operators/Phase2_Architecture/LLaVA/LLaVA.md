### 🧱 1. 为什么一定要加 `.contiguous()`？（显存视角的魔法）

这段代码的背景是自回归大模型的“错位预测（Shift）”：
```python
shifted_logits = logits[..., :-1, :].contiguous()
shifted_labels = labels[..., 1:].contiguous()
```

#### 🚨 灾难的起因：PyTorch 的“偷懒”机制 (View vs. Copy)
当你在 PyTorch 中对一个张量进行切片操作（比如 `[:-1]` 取除了最后一个之外的所有元素），**PyTorch 底层为了节省时间和显存，绝对不会去复制一份新的数据！**

它只会“偷懒”地修改这个张量的**元数据（Metadata，比如步长 Stride、偏移量 Offset 和形状 Shape）**。这被称为创建了一个**“视图（View）”**。

打个比方：
*   内存里原本存着 `[A, B, C, D]`（物理上是连续紧挨着的）。
*   你切片 `[1:]`，PyTorch 只是把读取指针往后挪了一格，告诉你现在是 `[B, C, D]`。但在物理内存里，它依然和 `A` 连在一起。
*   这种切片操作会导致张量在逻辑上是完整的，但在物理内存上变得**“不连续（Non-contiguous）”**。

#### 💥 为什么会报错崩溃？
切片完之后，我们通常要算交叉熵损失。为了匹配 `CrossEntropyLoss` 的输入要求，我们必须把 `shifted_logits` 从三维 `[Batch, Seq_Len, Vocab]` 展平（Flatten）成二维 `[Batch * Seq_Len, Vocab]`。

在 PyTorch 中，展平操作通常用 `.view(-1, vocab_size)`。
**但是，`.view()` 函数有一个极其严苛的底层死规定：它要求张量在物理内存中必须是 100% 连续的！**

如果你对一个切片后“不连续”的张量调用 `.view()`，PyTorch 会直接抛出著名的致命报错：
`RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.`

#### 🛠️ `.contiguous()` 的救场
当你在切片后加上 `.contiguous()` 时，底层发生了什么？
1.  PyTorch 检查一下当前张量的内存连不连续。
2.  如果不连续，它会**在显存里强行开辟一块全新的、干干净净的连续空间**。
3.  把切片后的数据，原封不动地**深拷贝（Deep Copy）**到这块新空间里。
4.  返回这个全新的、连续的张量。

这样，后续你再调用 `.view()` 或者送进 C++/CUDA 写的底层算子时，就绝对不会报错了。

**💡 面试话术：**
> “因为大模型自回归训练需要对 logits 和 labels 进行 Shift 切片操作，这破坏了张量在底层内存的连续性（Stride 发生变化）。为了满足后续 `.view()` 展平操作以及 CUDA 底层算子对连续内存的要求，必须调用 `.contiguous()` 强制进行内存重排和深拷贝，否则会引发 RuntimeError。”

---

### 🪣 2. `torch.full` 怎么用？（最优雅的填充工具）

在大模型指令微调（SFT）中，我们需要把不想计算 Loss 的地方（比如图片、用户的 Prompt）全部掩码掉，通常设为 `-100`。

这个时候，`torch.full` 就是你最好的帮手。

#### 📖 核心用法
`torch.full` 的作用非常纯粹：**创建一个指定形状（Shape）的张量，并把里面所有的元素都填满你指定的那个数字（Fill Value）。**

你可以把它看作是 `torch.zeros()`（全填 0）和 `torch.ones()`（全填 1）的**终极泛化版本**。

**基础语法：**
```python
torch.full(size, fill_value, dtype=None, device=None)
```

#### 💻 真实大模型代码演示

假设我们现在的 Batch Size 是 4，用户的 Prompt 长度是 128。我们需要生成一个形状为 `[4, 128]`，且里面全都是 `-100` 的掩码张量。

```python
import torch

B = 4
Prompt_Len = 128
IGNORE_INDEX = -100

# 用法演示
prompt_labels = torch.full(
    size=(B, Prompt_Len),       # 第一参数：元组形式的形状 (4, 128)
    fill_value=IGNORE_INDEX,    # 第二参数：要填入的数字 -100
    dtype=torch.long,           # 强烈建议指定类型！因为 Label 必须是整数 (LongTensor)
    device='cuda:0'             # 直接在 GPU 上创建，省去从 CPU 搬运的开销
)

print(prompt_labels.shape)  # 输出: torch.Size([4, 128])
print(prompt_labels[0, :5]) # 输出: tensor([-100, -100, -100, -100, -100], device='cuda:0')
```

#### ⚠️ 避坑指南（为什么要指定 `dtype=torch.long`？）
如果你只写 `torch.full((4, 128), -100)`，PyTorch 会根据 `-100` 自动推断类型。
但在某些旧版本或特定环境下，如果你填的是一个浮点数（比如 `torch.full((2,2), 3.14)`），它会默认生成 `float32`。
而大模型算 Loss 时，**分类任务的标签（Labels）必须是 64 位整数（`torch.long` / `torch.int64`）**。如果类型不对，传给 `CrossEntropyLoss` 时又会当场报错。所以，养成显式指定 `dtype=torch.long` 的好习惯，是资深工程师的标志。

---

### 🎯 总结

*   **`.contiguous()`**：显存整理大师。切片后必加，为了让张量在物理内存中排好队，防止后续展平操作（`.view`）崩溃。
*   **`torch.full()`**：批量填充大师。一键生成全 `-100` 的掩码张量，是大模型 SFT 阶段构造 Loss Masking 最优雅的 API。

掌握了这两个 API 的底层逻辑，你手撕的 PyTorch 代码就真正具备了**“工业级、可运行（Production-ready）”**的质感！