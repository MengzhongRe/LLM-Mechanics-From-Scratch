import torch
# 手动实现softmax函数
def my_softmax(x,dim=-1):
    # 将x减去每行最大值，防止经过指数运算后计算机内存溢出，NaN
    # 取完最大值之后与原张量X大小不匹配，所以需要keepdim=True
    # pytorch的max函数在输入dim参数后会返回一个(结果,结果索引)的元组，我们只要结果所以取第一个值
    max_val = torch.max(x,dim=dim,keepdim=True)[0]
    # 在pytorch中变量原地操作会释放原先变量内存，导致在反向传播时因丢失激活值梯度无法计算，因此在自定义算子时最好定义一个新变量
    x_safe = x - max_val
    exp_x = torch.exp(x_safe) # 计算指数
    sum_exp_x = torch.sum(exp_x,dim=dim,keepdim=True) # 在dim 维度计算指数和，同时保持维度以便与分子相除
    return exp_x / sum_exp_x
# =========================
# 测试用例
# =========================
if __name__ == '__main__':
    x = torch.tensor([[1000.0,2000.0,3000.0]])
    print(torch.softmax(x,dim=-1))
    print(my_softmax(x,dim=-1))
    assert torch.allclose(torch.softmax(x,dim=-1), my_softmax(x,dim=-1))
    print('验证通过，sofmtax实现成功')