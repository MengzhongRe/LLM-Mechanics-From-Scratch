# 手撕RoPE（旋转位置嵌入向量）

import torch

# =====================================================
# 模块一：预计算三角函数缓存
# ====================================================
def precompute_freqs_cos_sin(dim: int, end: int, theta: float = 10000.0):
    """
    预计算RoPE的cos,sin矩阵（公式34的实数派实现）
    参数：
        dim: 参与旋转的特征维度(rpoe_dim)
        end: 最大支持的序列长度(Seq_len)
        theta: 频率基数
    返回：
        cos, sin: 形状均为 [end,dim] 的浮点张量
    """
    # 1. 计算dim/2个复平面的角速度: \omega_i = 1 / (theta ** (2i / d))
    # 工业界标准写法： 利用torch.arange((0,dim,2)) 生成[0,2,4...,dim]
    freqs = 1.0 / (theta ** (torch.arange(0,dim,2,dtype=torch.float32) / dim))

    # 2. 生成绝对位置 t: [0,1,2,3...end - 1]
    t = torch.arange(end,dtype=torch.float32)

    # 3.矩阵外积： 计算每个位置、每个平面的旋转角度:m * \omega_i,0<=m<=end-1
    # t是一维张量[end],freqs是一维张量[dim/2] -> freqs_outer的形状为[end,dim/2]
    freqs_outer = torch.outer(t,freqs)

    # 4. 对齐维度：把[end,dim/2]复制拼接成[end,dim]
    # 这是为了和词向量的维度严格对其，方便后续作逐元素乘法
    freqs_outer = torch.cat([freqs_outer,freqs_outer],dim=-1)

    # 5. 生成对应的cos,sin张量，常驻内存，推理时直接查表
    cos = torch.cos(freqs_outer)
    sin = torch.sin(freqs_outer)

    return cos,sin

# ===========================================================
# 模块二： 交错翻转(Rorate_Half)
# ===========================================================
def rotate_half(x: torch.Tensor):
    """
    实现公式34中后半段的[-x2,x1]张量
    做法：将向量在最后一个维度dim=-1切成两半。前半部分放在后面，后半部分取负数放在前面
    """
    # chunk(2,dim-1)将向量在特征维度从中间切开，不发生显存复制，只改变视图
    x1,x2 = x.chunk(2,dim=-1)
    # 拼接： [-x2,x1]
    return torch.cat([-x2,x1],dim=-1)

# =============================================================
# 模块三：实现前向传播
# =============================================================
def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int = None):
    """
    应用RoPE.支持DeepSeek的解耦机制（只旋转部分维度）和混合精度保护
    x: [Batch_size,Seq_len,num_heads,head_dim]
    cos,sin: [Seq_len,rope_dim]
    rope_dim: 参与旋转的维度大小。为None则全部旋转
    """
    head_dim = x.shape[-1]
    if rope_dim is None:
        rope_dim = head_dim
    
    # 1. 解耦机制：切开张量
    x_rot = x[..., :rope_dim] # 需要接受旋转的维度
    x_pass = x[..., rope_dim:] # 不需要旋转的维度

    # 2.精度防爆：强制提升为fp32
    x_rot_fp32 = x_rot.float()

    # 3.将cos,sin的形状从[Seq-Len,rope_dim]广播为[1,Seq_len,1,rope_dim]
    # 以完美匹配x_rot_fp32的4D形状
    cos = cos.view(1,cos.shape[0],1,cos.shape[1]).to(x_rot_fp32.device)
    sin = sin.view(1,sin.shape[0],1,sin.shape[1]).to(x_rot_fp32.device)

    # 4. 计算公式34，实现实数矩阵的旋转
    x_rotated_fp32 = (x_rot_fp32 * cos) + (rotate_half(x_rot_fp32) * sin)

    # 5.还原现场：降级并拼接
    x_rotated = x_rotated_fp32.to(x.dtype)

    if rope_dim < head_dim:
        return torch.cat([x_rotated,x_pass],dim=-1)
    else:
        return x_rotated

# ====================================================================
# 测试用例
# =====================================================================
if __name__ == '__main__':
    # 1. 定义全局变量
    BATCH_SZIE = 2
    SEQ_LEN = 1024
    DIM = 4096
    NUM_HEADS = 32
    HEAD_DIM = DIM // NUM_HEADS
    ROPE_DIM = HEAD_DIM // 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    print('启动RoPE工业级算子！')

    # 2.构造输入数据
    q = torch.randn((BATCH_SZIE,SEQ_LEN,NUM_HEADS,HEAD_DIM),dtype=dtype,device=device)
    # 预计算三角函数
    cos,sin = precompute_freqs_cos_sin(dim=ROPE_DIM,end=SEQ_LEN,theta=10000.0)

    # 3.对输入数据进行旋转
    q_rotated = apply_rotary_emb(q,cos,sin,ROPE_DIM)

    print(f'[*] 输入Query数据形状为: {q.shape}')
    print(f'[*] 输出Query 形状: {q_rotated.shape}')
    print(f'cos,sin缓存形状: {cos.shape}')

    # 4.验证解耦机制是否生效
    diff_rotated = (q[..., :ROPE_DIM] - q_rotated[..., :ROPE_DIM]).abs().max().item()
    diff_pass = (q[..., ROPE_DIM:] - q_rotated[..., ROPE_DIM:]).abs().max().item()

    print('\n边界验证！')
    print(f' -> 旋转部分的变化量（前六十四维）: {diff_rotated:.4f} (期望 > 0)')
    print(f' -> 未旋转部分的变化量(后六十四维): {diff_pass:.4f}')

    if diff_pass == 0.0 and diff_rotated > 0.1:
        print('\n伟大的胜利！ 解耦RoPE成功！！！')
    else:
        print('失败，请检查代码！')





