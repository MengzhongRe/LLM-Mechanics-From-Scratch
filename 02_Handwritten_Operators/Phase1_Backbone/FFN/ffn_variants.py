# 手撕Transformer中前馈神经网络层(feed-forward network)的演进代码
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# 代码一：VanillaFFN(Attention is all you need的源代码骨架)
# ===========================================================
# 传统FFN需要对当前层输入进行升维度（一般为4倍），然后再降维
class VanillaFFN(nn.Module):
    def __init__(self,d_model: int, hidden_dim: int):
        super().__init__()
        # 1. 升维矩阵（知识的Key）
        self.w1 = nn.Linear(d_model,hidden_dim)
        # 2. 激活函数
        self.act = nn.ReLU()
        # 3. 降维矩阵(知识的Value)
        self.w2 = nn.Linear(hidden_dim,d_model)
        # 总参数量为2 * d_model * hidden_dim
    
    def forward(self,x: torch.Tensor):
        # 先升维再激活最后降
        return self.w2(self.act(self.w1(x)))