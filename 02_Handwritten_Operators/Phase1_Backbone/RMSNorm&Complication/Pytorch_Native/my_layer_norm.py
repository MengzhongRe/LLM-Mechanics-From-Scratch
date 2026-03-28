# 用Pytorch手动实现LayerNorm

import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self,dim: int,eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # \\gamma 缩放系数
        self.bias = nn.Parameter(torch.zeros(dim)) # \\beta 平移系数

    def forward(self,x):
        # 输入x: [Batch,Seq_len,dim]
        # 强制转换为float32,防止求和溢出
        x_fp32 = x.float()
        # 求均值mu = [B,L,1]
        mu = x_fp32.mean(dim=-1,keepdim=True)
        # 求差值
        diff = x_fp32 - mu
        # 求方差
        var = (diff ** 2).mean(dim=-1,keepdim=True)
        # 归一化并转回原类型
        # x_nomr: [B,L,dim]
        x_norm = (diff / torch.sqrt(var + self.eps)).to(x.dtype)
        # 归一化后乘以缩放系数和偏移量返回[B,L,dim]
        return x_norm * self.weight + self.bias

# ===========================================
# 测试用例
# ==========================================
if __name__ == '__main__':

    # 1.定义全局变量
    Batch_size = 2
    Seq_len = 1024
    Dim = 4096
    dtype = torch.bfloat16

    # 2.构造输入数据
    x = torch.randn((Batch_size,Seq_len,Dim),dtype=dtype)

    # 3.实例化手撕的LayerNorm和官方torch.nn.LayerNorm类
    my_layernorm = MyLayerNorm(Dim,eps=1e-5).to(dtype)
    official_layernorm = nn.LayerNorm(Dim).to(dtype)
    # 强制两者的权重和偏置相等
    official_layernorm.weight.data = my_layernorm.weight.data.clone()
    official_layernorm.bias.data = my_layernorm.bias.data.clone()
    # 4.执行测试
    print(f'[*] 测试开始...')
    print(f'\t输入x的维度: {x.shape}')

    y_my = my_layernorm(x)
    y_official = official_layernorm(x)

    print(f'\t我的输出维度: {y_my.shape},输出类型为: {y_my.dtype}')
    print(f'\t官方的输出维度为: {y_official.shape},输出类型为: {y_official.dtype}')

    max_diff = (y_my - y_official).abs().max().item()
    print(f'[*] 最大绝对误差: {max_diff}')

    assert torch.allclose(y_my,y_official,rtol=1e-2,atol=1e-2),'输出结果不一致，实现失败！'
    print(f'[*] 实现成功！')



# 关于为何LayerNorm传播中，需要先将输入数据x(fp16或bf16)转换为float32，然后在归一化后重新转换回x.dtype？

# 🧠 知识笔记：为什么算子内部需要 `fp32` 的反复横跳？

# 首先，我们要确立一个大前提：**现代大模型（如 LLaMA、DeepSeek）的主体权重和激活值，全都是用 `bf16` 或 `fp16`（16位浮点数）来存储和计算的。** 因为相比 `fp32`，它们能省下一半的显存，并且在 GPU 的 Tensor Core 上算矩阵乘法极快。

# 既然平时都用 16 位，为什么一到 LayerNorm / RMSNorm，就非要临时切成 32 位呢？

# ## 一、 为什么要先转成 `float32`？（为了“防爆”与“防吞”）

# 在 RMSNorm 中，最危险的一步操作是求均方根：**先平方，再沿 4096 个维度求和（`sum` / `mean`）。** 
# 这一步对于 16 位浮点数来说，是一场灾难。

# ### 灾难 1：FP16 的“直接溢出 (Overflow)”
# `fp16`（半精度浮点数）的最大表示范围非常小，它的上限是 **`65504`**。
# * 假设输入向量 $x$ 中的每个元素平均大小只有 `4.0`。
# * 第一步平方：$4.0^2 = 16.0$。
# * 第二步求和：大模型的隐藏维度（dim）通常是 4096。沿着这 4096 个维度求和：$16.0 \times 4096 = 65536$。
# * **Boom 💥！** $65536 > 65504$。
# * 你的计算结果瞬间变成了 `NaN`（Not a Number）或 `Inf`。然后这个 `NaN` 会随着网络像瘟疫一样传染，整个模型的输出全部变成乱码，训练当场崩溃！

# ### 灾难 2：BF16 的“大数吃小数 (Swamping)”
# 有人会问：*“那我现在都用 `bf16` 了，它的范围和 `fp32` 一样大（能到 $10^{38}$），不会溢出，总不用转了吧？”*
# * **错！** `bf16` 虽然范围大，但它为了扩大范围，牺牲了极其严重的**精度（尾数位只有 7 位）**。
# * `bf16` 只能精确表示大约 **2到3位十进制有效数字**。
# * 当你在做累加运算（求均值）时：假设累加器里已经攒了一个比较大的数（比如 1000），此时你要加上一个很小的数（比如 0.01）。
# * 在 `bf16` 的世界里，$1000 + 0.01 = 1000$！那个 `0.01` 因为精度不够，直接被当成空气**“吞掉”**了（这在数值分析里叫 Swamping / Rounding Error）。
# * 累加 4096 次，这种微小的舍入误差会疯狂累积。最终算出来的方差 $\sigma^2$ 会严重偏离真实值，导致归一化失效，模型完全无法收敛。

# **👉 结论：**
# 转成 `float32`，就像是给计算过程申请了一张**“无限大的高精度草稿纸”**。在求平方、求和这些对精度极其敏感的操作时，我们在草稿纸上安安全全、精精确确地算完。

# ---

# ## 二、 为什么算完又要转回原格式 `bf16`？（为了“全局规矩”与“性能”）

# 既然 `float32` 这么好，这么精确，我们干脆直接返回 `float32` 的结果不就行了？
# **绝对不行！**

# ### 1. 遵守整个大模型的“数据流规矩”
# Norm 层不是孤立的。它后面紧接着就是 Attention 层（QKV 的矩阵乘法）和 MLP 层（全连接网络）。
# * 这些庞大的线性层（Linear Layers），它们的权重全是 `bf16`。
# * 现代 GPU 的 Tensor Core（专门做矩阵乘法的物理硬件）如果发现输入是 `fp32`，就无法开启全速的 `bf16` 矩阵乘法加速，**前向传播速度会瞬间暴跌 2 到 4 倍！**

# ### 2. 拯救显存与访存带宽 (Memory Bound)
# * 整个大模型的前向传播，实质上就是把张量（Tensor）在 GPU 的显存（HBM）和计算核心（SRAM）之间来回搬运。
# * 如果你返回 `fp32`，这就意味着你输出的张量体积**大了一倍**。
# * 在推理（Inference）时，这些激活值要被存入 KV Cache。如果全变成 `fp32`，你的显卡本来能支持 100 个人同时并发聊天，现在只能支持 50 个人，老板直接让你毕业。


