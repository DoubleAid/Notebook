# torch.BatchNorm2d 功能介绍

BatchNorm2d（二维批量归一化）是PyTorch中用于卷积神经网络的重要层，主要用于加速训练、提高模型稳定性和泛化能力。

## 主要作用

1. 解决内部协变量偏移（Internal Covariate Shift）
2. 在深度网络中，随着训练进行，各层输入的分布会发生变化
3. 这导致后续层需要不断适应输入分布的变化，减慢训练速度
4. BatchNorm2d将每层的输入归一化到稳定的分布

## 标准化计算过程

对于卷积网络的每个特征通道：

### 计算批次统计量：对当前批次中该通道的所有值

```
μ = 均值(该通道所有激活值)
σ² = 方差(该通道所有激活值)
```

### 归一化

```
x_norm = (x - μ) / √(σ² + ε)  # ε是防止除零的小常数
```

### 缩放和平移

```
y = γ * x_norm + β
```

其中γ、β是可学习参数，让网络能够恢复原有的表达能力

## 具体参数和作用

```python
torch.nn.BatchNorm2d(
    num_features,               # 输入特征图的数量（通道数）
    eps=1e-5,                   # 数值稳定性常数
    momentum=0.1,               # 运行均值/方差的动量
    affine=True,                # 是否学习γ和β参数
    track_running_stats=True    # 是否跟踪运行统计量
)
```

## 工作模式

**训练模式**

+ 使用当前批次的统计量（μ, σ）进行归一化
+ 更新运行均值/方差（指数移动平均）
+ 包含可学习的γ、β参数

**评估模式**

+ 使用训练期间积累的运行统计量
+ γ、β参数固定
+ 不更新统计量

## 实际示例

```python
import torch
import torch.nn as nn

# 假设输入形状: (batch_size=32, channels=64, height=28, width=28)
batch_norm = nn.BatchNorm2d(num_features=64)

# 训练模式
batch_norm.train()
x_train = torch.randn(32, 64, 28, 28)
y_train = batch_norm(x_train)  # 使用批次统计量

# 评估模式
batch_norm.eval()
x_test = torch.randn(16, 64, 28, 28)
y_test = batch_norm(x_test)  # 使用运行统计量
```
