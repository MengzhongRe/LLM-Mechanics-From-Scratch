import torch
import math
import torch.nn as nn
# 手动实现多头掩蔽注意力
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout_p=0.1):
        super().__init__()
        assert d_model % num_heads == 0,"d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self,q,k,v,mask=None):
        #q,k,v: [Btach_size,Seq_len,d_model]
        # 先根据输入获取批量大小
        batch_size = q.size(0)
        # 先把q输入W_q得到Q:[batch_size,seq_len,d_model] -> [batch_size,seq_len,d_model]
        # 再用view函数分头-> [batch_size,seq_len,num_heads,d_k]
        # 再用tranpose函数调换seq_len 和 num_heads维度实现注意力头的并行计算-> [batch_size,num_heads,seq_len,d_k]
        Q = self.W_q(q).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        K = self.W_k(k).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        V = self.W_v(v).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        # 缩放点积注意力
        # scores需要除以根号下d_k,防止Q,K相乘之后方差从1扩大到d_k，再经过softmax后概率值过于尖锐导致梯度消失，学习失败
        # [batch_size,num_heads,seq_len,d_k] * [batch_size,num_heads,d_k,seq_len] -> [batch_size,num_heads,seq_len,seq_len]
        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)

        # 掩蔽注意力: 把带有mask标志的token赋值为无穷大然后经过softmax后概率值为0
        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)
        # 将scores送入softmax函数计算注意力权重
        attention_weights = torch.softmax(scores,dim=-1)
        attention_weights = self.dropout(attention_weights) # 对注意力权重正则化
        # 加权求和:[B,H,L,L] * [B,H,L,d_k] -> [B,H,L,d_k]
        out = torch.matmul(attention_weights,V)
        # 在将结果输入到线性层之前，需要对矩阵out合并注意力头
        # 在合并注意力头之前，需要先用contiguous函数，开辟新的内存空间存储数据以供view函数调整维度，否则会报错
        # [B,H,L,d_k] -> [B,L,H,d_k] -> [B,L,d_model]
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        # [B,L,d_model] -> [B,L,d_model]
        return self.W_o(out)

# ================================================
# 测试用例 (Test Case)
# ================================================
if __name__ == '__main__':
    print('开始测试Multi-Head Attention 模块...')

    # 1.设定超参数
    BATCH_SIZE = 2
    SEQ_LEN = 10
    D_MODEL = 512
    NUM_HEADS = 8

    # 2. 实例化模型
    mha = MultiHeadAttention(D_MODEL,NUM_HEADS,dropout_p=0.1)

    # 3. 伪造输入数据
    # 模拟两个句子，每个句子10个token,每个token512维
    q = torch.randn(BATCH_SIZE,SEQ_LEN,D_MODEL)
    k = torch.randn(BATCH_SIZE,SEQ_LEN,D_MODEL)
    v = torch.randn(BATCH_SIZE,SEQ_LEN,D_MODEL)

    # 4. 伪造因果掩码(Causal Mask)
    # 在GPT等Decoder架构中，当前词看不到未来词
    # 我们用torch.tril生成一个下三角矩阵：1表示允许看，0表示遮蔽
    # 形状:[SEQ_LEN,SEQ_LEN] -> [1,1,SEQ_LEN<SEQ_LEN]以便利用广播机制
    causal_mask = torch.tril(torch.ones(SEQ_LEN,SEQ_LEN)).view(1,1,SEQ_LEN,SEQ_LEN)
    print(f'生成的因果掩码矩阵为:\n{causal_mask[0,0]}')

    # 5. 前向传播测试
    try:
        # 传入q,k,v和mask
        output = mha(q,k,v,causal_mask)
        # 6. 维度检查
        assert output.shape == (BATCH_SIZE,SEQ_LEN,D_MODEL),f'形状错误，期望形状为 {(BATCH_SIZE,SEQ_LEN,D_MODEL)},\
            实际为： {output.shape}'
        print('\n 测试通过！(Test Passed!)')
        print(f'输入q,k,v 形状为: {q.shape}')
        print(f'输出 output 形状为: {output.shape}')
    except Exception as e:
        print(f'\n 测试失败！(Test Failed!)\n报错信息为:{e}')

