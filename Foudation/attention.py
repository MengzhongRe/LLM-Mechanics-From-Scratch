import torch
import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads == 0,"d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
    
    def forward(self,q,k,v,mask=None):
        #q,k,v: [Btach_size,Seq_len,d_model]
        # 先根据输入获取批量大小
        batch_size = q.size(0)
        # 先把q输入W_q得到Q:[batch_size,seq_len,d_model] -> [batch_size,seq_len,d_model]
        # 再用view函数分头-> [batch_size,seq_len,num_heads,d_k]
        # 再用tranpose函数调换seq_len 和 num_heads维度实现注意力头的并行计算-> [batch_size,num_heads,seq_len,d_k]
        Q = self.W_q(q).view(batch_size,-1,self.num_heads,self.k).transpose(1,2)
        K = self.W_k(k).view(batch_size,-1,self.num_heads,self.d_k).tranpose(1,2)
        V = self.W_v(v).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        # 缩放点积注意力
        # scores需要除以根号下d_k,防止Q,K相乘之后方差从1扩大到d_k，再经过softmax后概率值过于尖锐导致梯度消失，学习失败
        # [batch_size,num_heads,seq_len,d_k] * [batch_size,num_heads,d_k,seq_len] -> [batch_size,num_heads,seq_len,seq_len]
        scores = torch.matmul(Q,K.tranpose(-2,-1)) / math.sqrt(self.d_k)

        # 掩蔽注意力: 把带有mask标志的token赋值为无穷大然后经过softmax后概率值为0
        if mask:
            scores = scores.masked_fill(mask==0,-1e9)
        # 将scores送入softmax函数计算注意力权重
        attention_weights = torch.softmax(scores)
        # 加权求和:[B,H,L,L] * [B,H,L,d_k] -> [B,H,L,d_k]
        out = torch.matmul(attention_weights,V)
        # 在将结果输入到线性层之前，需要对矩阵out合并注意力头
        # 在合并注意力头之前，需要先用contiguous函数，开辟新的内存空间存储数据以供view函数调整维度，否则会报错
        # [B,H,L,d_k] -> [B,L,H,d_k] -> [B,L,d_model]
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        # [B,L,d_model] -> [B,L,d_model]
        return self.W_o(out)


import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout_p=0.1):
        super().__init__()
        assert d_model % num_heads == 0,'d_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self,q,k,v,mask=None):
        batch_size = q.size(0)

        Q = self.W_q(q).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        K = self.W_k(k).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        V = self.W_v(v).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)

        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)

        if mask is not None:# mask是pytorch张量，若不显示指定格式可能会报错
            scores = scores.masked_fill(mask==0,-1e9)
        # 
        attention_weights = torch.softmax(scores,dim=-1)# softmax函数最后需指定计算维度，一般为dim=-1
        attention_weights = self.dropout(attention_weights) # 加分项
        out = torch.matmul(attention_weights,V)

        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)


        return self.W_o(out)

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout_p=0.1):
        assert d_model % num_heads == 0,'d_model must be divisible by num_heads'

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

        self.dorpout = nn.Dropout(dropout_p)

    def forward(self,q,k,v,mask=None):
        batch_size = q.size(0)

        Q = self.W_q(q).view(batch_size,-1,self.num_heads,self.d_k).view(1,2)
        K = self.W_k(k).view(batch_size,-1,self.num_heads,self.d_k).view(1,2)
        V = self.W_v(v).view(batch_size,-1,self.num_heads,self.d_k).view(1,2)

        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)
        
        attention_weights = torch.softmax(scores,dim=-1)
        attention_weights = self.dorpout(attention_weights)

        out = torch.matmul(attention_weights,V)

        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        return self.W_o(out)

        