# 手动实现数值稳定，基于log_softmax的交叉熵损失函数,支持ignore_index功能，适用于大模型预训练和指令微调
import torch

def safe_cross_entropy_loss(logits,labels,ignore_index=-100):
    """
    :param logits:[Batch_size * Seq_len,Vacab_size]
    :param labels:[Batch_size * Seq_len]
    """
    # ======================================
    # 1.找出有效token，丢弃ignore_index的token
    # ======================================
    valid_mask = (labels != ignore_index)

    # 仅提取有效的logits和labels，有效节省计算量
    # valid_logits: [Num_valid,vacab-size]
    # valid_labels: [Num_valid]
    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]

    # 如果全是-100。直接返回0梯度
    if valid_labels.numel() == 0:
        return torch.tensor(0.0,device=logits.device,requires_grad=True)
    
    # ======================================
    # 2.减去最大值，防止e^x溢出
    # ======================================
    # 沿着Vocab维度找最大值，keepdim=True,保持维度不变，方便后续计算
    max_logits,_ = torch.max(valid_logits,dim=-1,keepdim=True)

    # 减去最大值，得到数值稳定的logits
    safe_logits = valid_logits - max_logits

    # ======================================
    # 3.计算Log_Softmax:x - log(sum(e^x))
    # 算指数和
    sum_exp = torch.sum(torch.exp(safe_logits),dim=-1,keepdim=True)
    # 对指数和求对数
    log_sum_exp = torch.log(sum_exp)
    # 得到每个词的log概率
    log_probs = safe_logits - log_sum_exp
    
    # ======================================
    # 4.根据标签索引提取对应token的log概率
    # ======================================
    # 用valid_labels作为索引，把正确答案对应的log_prob抠出来
    # gather函数：沿着dim维度，根据index索引提取元素，unsqueeze(-1)把valid_labels从[Num_valid]变成[Num_valid,1]，方便gather操作
    target_log_probs = log_probs.gather(dim=-1,index=valid_labels.unsqueeze(-1)).squeeze(-1)
    # squeeze(-1)把结果从[Num_valid,1]变回[Num_valid]，方便后续计算损失均值

    # ======================================
    # 5.计算交叉熵损失，取负号，并求均值
    # ======================================
    loss = -target_log_probs.mean()

    return loss

if __name__ == '__main__':
    V = 10000

    logits = torch.randn(5,V) * 100 # 故意放大制造溢出风险
    labels = torch.tensor([10,-100,200,-100,9999])

    my_loss = safe_cross_entropy_loss(logits,labels,ignore_index=-100)
    hf_loss = torch.nn.functional.cross_entropy(logits,labels,ignore_index=-100)

    print(f'My loss: {my_loss.item():.4f}')
    print(f'HF Loss: {hf_loss.item():.4f}')
    assert torch.allclose(my_loss,hf_loss),'计算错误，请检查实现细节！'