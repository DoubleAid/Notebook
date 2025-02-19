# 以 CRFAR10 model 为例
![label](https://www.researchgate.net/profile/Yiren-Zhou-6/publication/312170477/figure/fig2/AS:448817725218817@1484017892180/Structure-of-CIFAR10-quick-model.png)

## 网络模型
```python
import torch
from torch.nn import Conv2d, Linear, MaxPool2d, Flatten
from torch.utils.tensorboard import SummaryWriter

class CIFAR10(torch.nn.Module):
  def __init__(self):
    super(CIFAR10, self).__init__()
    # 输入 3@32x32 输出 32@32x32, padding 为 2
    self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size = 5, padding = 2)
    self.maxpool1 = MaxPool2d(2)
    self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
    self.maxpool2 = MaxPool2d(2)
    self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
    self.maxpool3 = MaxPool2d(2)
    self.flatten = Flatten()
    self.linear1 = Linear(in_features=1024, out_features=64)
    self.linear2 = Linear(in_features=64, out_features=10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.maxpool2(x)
    x = self.conv3(x)
    x = self.maxpool3(x)
    x = self.flatten(x)
    x = self.linear1(x)
    x = self.linear2(x)
    return x

test_model = CIFAR10()
print(test_model)
input = torch.ones((64, 3, 32, 32))
output = test_model(input)
print(output.shape)

writer = SummaryWriter("../logs/test3")
writer.add_graph(test_model, input)
writer.close()
```

## 使用 sequential
和上面的相差不大， 主要是更加简介
```python
import torch
from torch.nn import Conv2d, Linear, MaxPool2d, Flatten
from torch.utils.tensorboard import SummaryWriter

class CIFAR10(torch.nn.Module):
  def __init__(self):
    super(CIFAR10, self).__init__()
    self.model = Sequential(
      # 输入 3@32x32 输出 32@32x32, padding 为 2
      Conv2d(in_channels=3, out_channels=32, kernel_size = 5, padding = 2)
      MaxPool2d(2)
      Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
      MaxPool2d(2)
      Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
      MaxPool2d(2)
      Flatten()
      Linear(in_features=1024, out_features=64)
      Linear(in_features=64, out_features=10)
    ）

  def forward(self, x):
    x = self.model(x)
    return x

test_model = CIFAR10()
print(test_model)
input = torch.ones((64, 3, 32, 32))
output = test_model(input)
print(output.shape)

writer = SummaryWriter("../logs/test3")
writer.add_graph(test_model, input)
writer.close()
```
