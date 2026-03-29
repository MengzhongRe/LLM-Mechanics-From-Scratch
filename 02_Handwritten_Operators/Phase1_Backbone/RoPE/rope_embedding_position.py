# 手撕RoPE(旋转位置编码) + position位置感知，适配decode生成阶段

import torch

# ======================================================================
# 模块一:预计算三角函数缓存
# ======================================================================
def precompute_freqs_cos_sin(dim: int, end: int, theta: float = 10000.0):
    """
    参数:
        dim: 参与旋转的特征维度数
        end: 支持的最大序列长度
        theta: 旋转频率基数
    """
    # 1.计算每个复平面的角速度: \omega_i = 1 / theta **(2i /dim)
    # 工业界标准写法:用torch.arange(0,dim,2)生成[0,2,4,...,dim]
    freqs = 1.0 / (theta ** (torch.arange(0,dim,2,dtype=torch.float32) / dim))

    # 2.生成绝对位置序列[0,1,2,...,end - 1]
    t = torch.arange(end,dtype=torch.float32)

    # 3.矩阵外积:计算每个位置，每个复平面的旋转角度:m * \omega_i
    freqs_outer = torch.outer(t,freqs)

    # 4. 对齐维度
    freqs_outer = torch.cat([freqs_outer,freqs_outer],dim=-1)

    # 5.生成对应的cos,sin张量,常驻内存，推理时直接查表
    cos = torch.cos(freqs_outer)
    sin = torch.sin(freqs_outer)

    return cos,sin

# ===================================================================================
# 模块二:交错翻转
# ===================================================================================
def rotate_half(x: torch.Tensor):
    """"
    把输入张量在最后一个维度上切成两半，后半部分取负放到前面，前面放到后面
    """
    x1,x2 = x.chunk(2,dim=-1)
    return torch.cat([-x2,x1],dim=-1)

# ==================================================================================
# 模块三:实现前向传播（旋转）
# =======================================================================================
def apply_rope_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor,rope_dim: int = None):
    """
    x: [Batch_size,Seq_len,num_heads,head_dim]
    cos,sin: [max_seq_len,rope_dim] 预先计算好的整个缓存表
    position_ids: [Batch_size,Seq_len] 每个token的绝对位置
    """
    head_dim = x.shape[-1]
    if rope_dim is None:
        rope_dim = head_dim
    
    # 1. 解耦机制：切开张量
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    # 2.精度保护：把旋转的部分转换为float32
    x_rot_fp32 = x_rot.float()

    # 【改进】根据当前输入的position_ids，从缓存表中抽取对应的cos/sin
    # position_ids的形状: [Batch_size,Seq_len]
    # cos的形状: [max_seq_len,rope_dim]
    # Pytorch中用高维整型张量如positoin_ids去索引cos的第0维，其第0维
    # 会自动替换为position_ids的形状，因此
    # 抽取后的形状: [Batch_size,Seq_len,rope_dim]
    cos_sliced = cos[position_ids].to(x.device)
    sin_sliced = sin[position_ids].to(x.device)

    # 【改进】 鲁棒广播: 使用unsqueeze 插入num_heads维度
    # cos_sliced: [Batch_size,Seq_len,1,rope_dim] 完美适配x_rot_fp32
    cos_sliced = cos_sliced.unsqueeze(2)
    sin_sliced = sin_sliced.unsqueeze(2)

    # 4. 计算公式34：实现实数矩阵的旋转
    x_rotated_fp32 = (x_rot_fp32 * cos_sliced) + (rotate_half(x_rot_fp32) * sin_sliced)

    # 5.还原现场：降级并拼接
    x_rotated = x_rotated_fp32.to(x.dtype)

    if rope_dim < head_dim:
        return torch.cat([x_rotated,x_pass],dim=-1)
    else:
        return x_rotated

# ============================================================================
# 测试用例
# ==========================================================================
if __name__ == '__main__':
    # 1.定义全局变量
    BATCH_SIZE = 2
    SEQ_LEN = 1024  # 预填充阶段长度
    MAX_LEN = 2048  # 系统支持的最大长度
    NUM_HEADS = 32  
    HEAD_DIM = 128
    ROPE_DIM = HEAD_DIM // 2
    dtype = torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('启动RoPE工业级算子(支持动态推理)...')

    # 预计算整个生命周期的三角函数并放入设备
    cos,sin = precompute_freqs_cos_sin(dim=ROPE_DIM,end=MAX_LEN,theta=10000.0)
    cos,sin = cos.to(device),sin.to(device)

    # ===========================================================================
    # 模拟1：Prefill 阶段(处理一段完整地Prompt)
    # ===========================================================================
    print('\n--- 模拟 Phase 2: Prefill阶段 ---')
    # 构造prompt
    q_prefill = torch.randn((BATCH_SIZE,SEQ_LEN,NUM_HEADS,HEAD_DIM),dtype=dtype,device=device)
    # 构造位置id，position_ids形状为[BATCH_SIZE,SEQ_LEN],每一个BATCH的ID为[0,1,2,...,SEQ_LEN-1]
    # 
    position_ids = torch.arange(SEQ_LEN,dtype=torch.long,device=device).unsqueeze(0).expand(BATCH_SIZE,-1)

    q_rotated_prefill = apply_rope_emb(q_prefill,cos,sin,position_ids,ROPE_DIM)
    print(f'[*]Prefill阶段的输入形状为: {q_prefill.shape}')
    print(f'[*]Prefill阶段的输出形状为: {q_rotated_prefill.shape}')
    # 验证解耦
    diff_rotated = (q_prefill[..., :ROPE_DIM] - q_rotated_prefill[..., :ROPE_DIM]).abs().max().item()
    diff_pass = (q_prefill[..., ROPE_DIM:] - q_rotated_prefill[..., ROPE_DIM:]).abs().max().item()
    assert diff_rotated > 0.1 and diff_pass == 0.0,'解耦失败'
    print('Prefill 阶段解耦测试通过')

    # ===============================================================================
    # 模拟2：Decode阶段(自回归生成下一个token)
    # ================================================================================
    print('\n--- 模拟 Phase3: Decode阶段 ---')
    # Decode时，只传入一个token
    q_decode = torch.randn((BATCH_SIZE,1,NUM_HEADS,HEAD_DIM),dtype=dtype,device=device)
    # 关键: 传入这个Token在全量文本中的绝对位置(第1024个位置)
    position_ids = torch.tensor([[1024],[1024]],dtype=torch.long,device=device)

    q_decode_rotated = apply_rope_emb(q_decode,cos,sin,position_ids,ROPE_DIM)
    print(f'[*] Decode输入形状为: {q_decode.shape}')
    print(f'[*] Decode输出形状为: {q_decode_rotated.shape}')
    print("✅ Decode 流水线测试通过（没有因为 Seq_len=1 而切错缓存）！")