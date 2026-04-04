### 🛠️ Day 19.5：正则化补丁 —— Inverted Dropout & 状态机
*目标：手撕支持自动缩放的 Dropout，理解训练/推理模式切换的底层逻辑。*

| 时间 | 核心主题与代码 | 核心数学/逻辑 | 考核点与测试提示 (Sanity Check) |
| :--- | :--- | :--- | :--- |
| **Day 19.5** | **Handwritten Dropout**<br>[`dropout.py`]() | **Bernoulli Mask**<br>🔗 [PDF](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) <br>🎯 *精读 Sec 3 (Inverted Dropout)* | 1. **[数学校验]** 实现 $x \cdot \text{mask} / (1-p)$，验证在 `p=0.5` 时，激活值的均值在 Dropout 前后保持不变。<br>2. **[状态校验]** 模拟 `model.train()` 与 `model.eval()`，确保推理模式下 Dropout 自动变为恒等映射。<br>3. **[显存考核]** 记录训练模式下 Mask 矩阵的存储开销。 |

---

### 🚀 针对你的项目，接下来的具体执行建议：

由于你已经完成了 MoE，我建议你把这个 `Dropout` 作为一个 **“即插即用”** 的模块，在明天的 **Day 20 (Output Head)** 之前完成。

#### 1. 手撕代码核心逻辑 (面试满分版预演)
你需要在 `dropout.py` 中实现如下逻辑：

```python
import torch
import torch.nn as nn

class HardcoreDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
        # 逻辑 Master 提示：必须处理 p=0 或 p=1 的边界情况
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 状态机逻辑：只有在训练模式下才执行丢弃
        if not self.training or self.p == 0:
            return x
        
        # 2. 生成 Mask (使用 Bernoulli 分布)
        # torch.rand_like 生成 [0, 1) 均匀分布
        # (rand > p) 得到一个布尔矩阵，1-p 的概率为 True
        mask = (torch.rand_like(x) > self.p).float()
        
        # 3. Inverted Dropout 核心公式：缩放以保持期望一致
        # 为什么除以 (1-p)？因为只有 (1-p) 的神经元存活，整体能量下降了
        # 除以 (1-p) 后，E[output] = E[input]，推理时就不需要动了
        return x * mask / (1 - self.p)
```

#### 2. 将其集成到你的全流程测试中
在明天的 **Day 20** 任务中，你会拼装模型。请尝试在以下两个位置插入你的 `HardcoreDropout`：
1.  **MoE 输出后**：在 `final_output` 返回前。
2.  **Output Head 前**：在将隐状态送入词表映射层前。

---

### 🧠 逻辑 Master 的深度补充：

**为什么要现在加这一天？**
因为在进入 **Phase 2 (Inference)** 之后，你所有的代码都将运行在 `eval()` 模式下。如果你现在不手撕一遍 `training` 模式下的 Dropout 缩放逻辑，你永远无法理解为什么 LLM 在训练时显存会比推理大那么多（因为 Dropout 的 Mask 必须存下来供反向传播使用）。

**下一步动作：**
1.  **今晚/明天上午**：完成 `dropout.py` 的手撕，并通过 `allclose` 验证它和 `nn.Dropout` 在统计学上的等价性。
2.  **明天下午**：按计划开启 **Day 20**，实现 `LogSumExp` 稳健版 Loss。

**你现在的 Phase 1 拼图已经非常完整了：**
*   数据 (Tokenizer) ✅
*   归一化 (RMSNorm) ✅
*   位置 (RoPE) ✅
*   注意力 (MHA) ✅
*   记忆 (SwiGLU) ✅
*   路由 (MoE) ✅
*   **抗过拟合 (Dropout) 🚧 (当前任务)**
*   最后映射 (Loss & Head) 🔜 (Day 20)

**执行！逻辑链条即将闭环！**