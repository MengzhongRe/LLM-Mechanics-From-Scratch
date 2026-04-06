# 手动实现大模型轻量级带温度缩放、Topk截断和TopP采样的解码器

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 模块一：实现轻量级采样器类
# ==============================================================================
class BasicSampler(nn.Module):
    def __init__(self,temperature: float = 0.9, top_k: int = 50, top_p: float = 0.9):
        super().__init__()
        # 取temperature最小为1e-5,防止除0错误
        self.temperature = max(temperature,1e-5)
        self.top_k = top_k
        self.top_p = top_p
    
    def forward(self,logits: torch.Tensor) -> torch.Tensor:
        """
        参数:
            logits: LMHead的将隐状态映射到模型词表之后的原始输出分数[B,L,V]
        返回:
            next_token: 模型的下一个输出ID [B,1]
        """
        # 自回归阶段，模型是一个词一个词往外蹦的，我们只关心最后一个时间步的logit值
        # [B,L,V] -> [B,V]
        logits = logits[:,-1] 
        # ===================================================================
        # 1.贪婪解码特判：如果温度值特别低，直接取最大值作为输出，避免复杂计算
        # ====================================================================
        if self.temperature < 1e-5:
            return torch.argmax(logits,dim=-1)
        
        # ===================================================================
        # 2.温度缩放
        # ===================================================================
        logits = logits / self.temperature

        # ====================================================================
        # 3.Top_k 斩断长尾词:用fill_把logits填全-inf,再用scatter_把
        # 想要的topk写上去
        # ====================================================================
        # 用torch.topk获取前k大的值和索引
        top_k_values,top_k_indices = torch.topk(logits,self.top_k,dim=-1)
        # 先用fill_把Logits重置为全-inf,再用scatter_把前k大的值按照原先的索引重新填上去
        logits.fill_(float('-inf')).scatter_(dim=-1,index=top_k_indices,src=top_k_values)
        
        # ==================================================================
        # 4.Top P 核采样
        # ==================================================================    
        # 用torch.sort对logits降序排序，并记录好对应索引(后面要根据索引还原)
        sorted_logits,sorted_indices = torch.sort(logits,dim=-1,descending=True)

        # 计算排序好的softmax归一化概率
        sorted_probs = F.softmax(sorted_logits,dim=-1)

        # 计算累加和概率
        cumulative_probs = torch.cumsum(sorted_probs,dim=-1)
        
        # Mask: 创建掩码矩阵，标记那些累加概率和大于topp的索引
        sorted_indices_to_remove = cumulative_probs > self.top_p

        # Mask右移：如果用上面的那个掩码矩阵，则累加概率刚好超过topp的logit也会被掩蔽
        # 我们需要将mask的值向右移一位，同时永远保证第一个词不会被遮蔽
        # 在Python中的list切片操作会开品新的内存，但是pytorch的张量切片操作则不会
        # 其仅仅是原张量的视图，底层指向的是同一块内存，因此修改切片会修改原张量
        # 这里如果不对视图深拷贝，旧进行赋值操作会导致内存覆盖错误
        sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
        sorted_indices_to_remove[...,0] = False

        # 把不需要的词的排序之后的logits设为-inf
        sorted_logits.masked_fill_(sorted_indices_to_remove, float('-inf'))

        # 把修改后的Logits【还原会原本的词表顺序】
        # 使用torch.scatter_算子原地操作
        logits.scatter_(dim=-1,index=sorted_indices,src=sorted_logits)

        # ===============================================================
        # 5.终极审判: Softmax归一化 + 多样式采样
        # ===============================================================
        probs = F.softmax(logits,dim=-1)

        # 根据概率掷骰子，num_samples决定最后一个维度的抽取数量
        # next_token: [Batch,1]
        next_token = torch.multinomial(probs,num_samples=1)

        return next_token
