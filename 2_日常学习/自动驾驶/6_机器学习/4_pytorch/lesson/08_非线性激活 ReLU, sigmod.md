# 非线性激活

主要使用 sigmod 和 ReLU

添加非线性特征， 增加

```python
import torch
from torch import nn

input = torch.tensor([[1, -0.4],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))

class ReLUTest(nn.Module):
  def __init__(self):
    super(ReLU, self).__init__()
    self.ReLU = nn.ReLU()

  def forward(self, x):
    x = self.ReLU(x)
    return x

relu_test = ReLUTest()
output = relu_test(input)
print(output)
```
