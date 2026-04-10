# 手动实现LLaVA 1.0,VLM(Vision-Language Model)视觉语言模型个的前向传播函数

import torch
import torch.nn as nn

class LLaVA_Alignment(nn.Module):
    def __init__(self,vision_dim=1024,llm_dim=4096,vocab_size=32000):
        super().__init__()
        # 1.核心算子:投影层
        # LLaVA 1.0是单层Linear,LLaVA 1.5是两层MLP（Linear -> GELU -> Linear）
        self.vocab_size = vocab_size
        self.projector = nn.Linear(vision_dim,llm_dim)
        # 2.LL M的词嵌入层(Word Embedding)
        self.text_embedding = nn.Embedding(vocab_size,llm_dim)
        # 3.模拟LL M的主干网络
        self.backbone = nn.Identity()
        # 4.模拟LLM的输出预测头
        self.lm_head = nn.Linear(llm_dim,vocab_size)
        # 5.定义忽略计算Loss 的index
        self.IGNORE_INDEX = -100
    
    def forward(self,image_features,prompt_ids,answer_ids):
        """
        :param image_features: [Batch_size,Num_Patches,vision_dim]
        :param prompt_ids: [Batch_size,Prompt_len]
        :param answer_ids: [Batch_size,Answer_len]
        """
        B = image_features.size(0)

        # =================================
        # 1.视觉特征对齐
        # =================================
        # [Batch_size,Num_Patches,vision_dim] -> [Batch_size,Num_Patches,llm_dim]
        image_embeds = self.projector(image_features)

        # =================================
        # 2.文本特征对齐
        # =================================
        # shape: [Batch_size,prompt_len,llm_dim]
        prompt_embeds = self.text_embedding(prompt_ids)
        # shape: [Batch_size,answer_len,llm_dim]
        answer_embeds = self.text_embedding(answer_ids)

        # ================================
        # 3.多模态拼接(Multimodal Concatenation)
        # ================================
        # 把图片、问题、答案嵌入拼接成完整的长序列
        # [Batch_size,Num_Patches + prompt_len + answer_len,llm_dim]
        input_embeds = torch.cat([image_embeds,prompt_embeds,answer_embeds],dim=1)

        # 送入大模型主干，得到隐藏状态
        hidden_states = self.backbone(input_embeds)

        # 得到词表的概率分布
        # shape:[Bacth_size,Seq_len,vocab_size]
        logits = self.lm_head(hidden_states)

        # ===============================
        # 4.构造标签
        # ===============================
        Num_Patches = image_embeds.size(1)
        Prompt_len = prompt_embeds.size(1)
        Answer_len = answer_embeds.size(1)

        # 4.1 屏蔽图片的loss，即把图片标签全部设为-100
        # shape: [Batch_size,Num_Patches]
        image_labels = torch.full((B,Num_Patches),self.IGNORE_INDEX,dtype=torch.long,device=image_features.device)
        # 4.2 屏蔽掉问题的loss
        # shape: [B,Prompt_len]
        prompt_labels = torch.full((B,Prompt_len),self.IGNORE_INDEX,dtype=torch.long,device=prompt_ids.device)
        # 4.3 保留原本的answer,即answer_ids
        # shape: [B,answer_len]
        answer_labels = answer_ids
        
        # 4.4 拼接完整的labels
        # shape: [B,Seq_len]
        labels = torch.cat([image_labels,prompt_labels,answer_labels],dim=1)

        # ================================
        # 5 计算损失
        # ================================
        # 由于自回归模型在t位置的输出实际上预测的是t + 1个位置的token，所以我们需要对labels和logits进行移位
        # logits要去掉最后一个
        shifted_logits = logits[...,:-1,:].contiguous()
        shifted_labels = labels[...,1:].contiguous()

        # 初始化loss函数
        loss_function = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        # 为了匹配crossentropyloss的输入形状要求
        # 要求为logits: [Batch_size * Seq_len,Vocab_size],labels: [Batch_size * Seq_len]
        # 因此必须用view函数调整
        loss = loss_function(shifted_logits.view(-1,self.vocab_size),shifted_labels.view(-1))

        return loss,logits

# ====================================
# 测试用例
# ===================================
if __name__ == '__main__':
    # 1.超参数设置
    # 模型参数设置
    vision_dim = 1024
    llm_dim = 4096
    vocab_size = 32000

    # 输入数据参数设置
    Batch_size = 2
    Num_Patches = 256
    Prompt_len = 200
    Answer_len = 800

    # 2.初始化模型
    llava = LLaVA_Alignment(vision_dim,llm_dim,vocab_size)
    # 3. 输入数据制造
    
    # 3.1 视觉特征输数据：连续的浮点数
    # 模拟CLIP VIT 的输出
    image_features = torch.randn((Batch_size,Num_Patches,vision_dim))

    # 3.2 prompt_dis:离散的整数token id（Long Tensor）
    # 范围必须在[0,vocab_size - 1]

    # torch.randint(row,high,size)用于生成最小值为low,最大值为high-1之间（所有概率相等）的形状为size的整数张量
    # torch.randint()经常被用于随机生成token id或labels 
    prompt_ids = torch.randint(low=0,high=vocab_size,size=(Batch_size,Prompt_len),dtype=torch.long)

    # 3.3 answer_ids:也是离散的整数token id(Long Tensor)
    answer_ids = torch.randint(low=0,high=vocab_size,size=(Batch_size,Answer_len),dtype=torch.long)

    # ==========================================
    # 4. 开始执行前向传播测试
    # ==========================================
    print('开始测试LLaVA前向传播...')
    print(f'输入Image shape: {image_features.shape}')
    print(f'输入Prompt_ids shape: {prompt_ids.shape}')
    print(f'输入Answer_ids shape: {answer_ids.shape}')
    print('-' * 40)

    loss,logits = llava(image_features,prompt_ids,answer_ids)

    Seq_len = Num_Patches + Prompt_len + Answer_len

    print(f'logits的维度为： {logits.shape}')
    print(f'logits期望输出维度为: [{Batch_size},{Seq_len},{vocab_size}]')
    print(f'loss为: {loss.item():.4f}!')

    assert logits.shape == (Batch_size,Seq_len,vocab_size),'logits 维度错误'
    assert not torch.isnan(loss),'loss 是NaN，发生了数值溢出'

    print('-' * 40)
    print('测试通过！')
