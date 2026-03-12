import torch
import torch.nn.functional as F

def generate_next_token(logits,temperature=1.0,top_k=50,top_p=0.9):
    """
    logits:[B,vocab_size]
    """
    if temperature < 1e-4:
        return torch.argmax(logits,dim=-1,keepdim=True) # 返回[B,1]

    # =======================
    # 1. 温度缩放
    # =========================
    logits = logits /temperature

    # =============================
    # 2. top_k长尾截断
    # ==============================
    # top_k取其和词表大小的最小值，防止因输入值超过词表大小而报错
    top_k = min(top_k,logits.size(-1))
    if top_k > 0:
        # 用torch.topk函数挑选出logits中最后一个维度前top_k大的值
        # torch.topk返回(值，索引)元组，值列表会自动按降序返回我们只取第一个
        top_values,_ = torch.topk(logits,top_k,dim=-1)


