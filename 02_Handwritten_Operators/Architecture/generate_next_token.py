import torch
import torch.nn.functional as F

def generate_next_token(logits, temperature=1.0, top_k=50, top_p=0.9):
    """
    大模型标准解码管线 (Interview Level)
    
    参数:
    logits: [Batch_size, vocab_size] 语言模型头输出的原始得分
    temperature: 浮点数，控制随机性
    top_k: 整数，保留前 k 个最高概率的词
    top_p: 浮点数，保留累加概率刚刚超过 p 的词 (核采样)
    
    返回:
    next_token: [Batch_size, 1] 抽样出的下一个词的 ID
    """
    
    # ---------------------------------------------------------
    # 1. 贪婪解码特判 (如果温度极低，直接选最大值，省去复杂计算)
    # ---------------------------------------------------------
    if temperature < 1e-5:
        # [Batch, 1]
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # ---------------------------------------------------------
    # 2. 温度缩放 (Temperature Scaling)
    # ---------------------------------------------------------
    logits = logits / temperature
    
    # ---------------------------------------------------------
    # 3. Top-K 截断 (剔除长尾低频词)
    # ---------------------------------------------------------
    top_k = min(top_k,logits.size(-1))
    if top_k > 0:
        # 找出第 k 大的值和对应的索引，torch.topk为自动将返回的结果降序排序
        # top_values:[Batch, k]
        top_values, _ = torch.topk(logits, top_k, dim=-1)
        
        # 取出每个 Batch 第 k 大的分数 (即 top_values 的最后一列)
        # kth_values: [Batch, 1]
        # 用[:,-1:]而非[:,-1]是因为后者会导致丢失最后一层的维度[B]，而前者会保留[B,1]
        # 效果类似于keepdim=True,以便和后面的logits[B，vocab_size]比较大小
        kth_values = top_values[:, -1:]
        
        # 核心逻辑：把所有严格小于第 k 大分数的 Logits，全部设为负无穷
        # torch.where函数作用torch.where(condition,x,y)，把符合条件的设为x,否则设为y
        logits = torch.where(logits < kth_values, torch.tensor(-float('inf'), device=logits.device), logits)
        # logits = logits.masked_fill(logits < kth_value,float('-inf'))
    # ---------------------------------------------------------
    # 4. Top-P 核采样 (大厂极其高频手撕考点！！！)
    # ---------------------------------------------------------
    if 0.0 < top_p < 1.0:
        # 步骤 A：把 Logits 降序排列 (记住原来的索引，后面要还原)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # 步骤 B：计算排序后的 Softmax 概率
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        # 步骤 C：计算累加概率 (Cumulative sum)
        # 比如 probs =[0.5, 0.3, 0.1, 0.1] -> cumulative =[0.5, 0.8, 0.9, 1.0]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 步骤 D：创建 Mask，标记那些累加概率超过 top_p 的位置
        # 比如 top_p = 0.85，那么 mask = [False, False, True, True]
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 🚨 【面试官疯狂加分项：Mask 右移魔术】🚨
        # 痛点：如果直接用上面的 Mask，那个刚好让累加值超过 p 的词也会被删掉！
        # 比如 0.5 + 0.3 = 0.8 < 0.85。下一个加了 0.1 变成 0.9 > 0.85，这个 0.1 的词必须保留！
        # 解法：把 Mask 向右平移一位，并且强行保证第 0 个词（概率最大的词）永远不被删！
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 步骤 E：把不需要的词的排序后 Logits 设为负无穷
        sorted_logits.masked_fill_(sorted_indices_to_remove, -float('inf'))
        
        # 步骤 F：把修改后的 Logits 【还原回原本的词表顺序】
        # 使用 scatter_ 算子：按照 sorted_indices，把 sorted_logits 填回空张量中
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        
    # ---------------------------------------------------------
    # 5. 终极审判：Softmax 归一化 + 多项式采样
    # ---------------------------------------------------------
    # 此时，那些被设为 -inf 的位置，概率已经完美变成了 0
    probs = F.softmax(logits, dim=-1)
    
    # 根据概率权重掷骰子，抽取 1 个 Token（num_samples决定最后一个维度的抽取数量）
    # next_token:[Batch, 1]
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token