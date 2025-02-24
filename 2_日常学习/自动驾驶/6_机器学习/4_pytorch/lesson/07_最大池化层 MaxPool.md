# Pooling Layers

+ MaxPool: 最大池（下采样）
+ MaxUnpool: (上采样)

最大池化的目的： 减小数据量， 保留特征， 加快训练速度

## Parameters

+ kernel_size:
+ stride： 默认和 kernel_size 相同
+ padding
+ dilation
+ return_indices (bool):
+ ceil_mode (bool): ceil 取上限 floor 取下线， 默认为 true， 当池化核不能完全覆盖时， 也会进行池化

## shape

+ Input: $(N, C, H_{in}, W_{in})$ or $(C, H_{in}, W_{in})$
+ Output: $(N, C, H_{out}, W_{out})$ or $(C, H_{out}, W_{out})$
  + $H_{out}=[{{H_{in}+2*padding[0]-dilation[0]*(kernelsize[0]-1)} \over stride[0]} + 1]$
  + $W_{out}=[{{W_{in}+2*padding[1]-dilation[1]*(kernelsize[1]-1)} \over stride[1]} + 1]$

```python
import torch
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))

class MyPool(torch.nn.Module):
  def __init__(self):
    super(MyPool, self).__init__()
    self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, ceil_mode=True)
  
  def forward(self, x):
    x = self.maxpool(x)
    return x

mypool = MyPool()
output = mypool(input)
```

```python
import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataload = torch.utils.data.DataLoader(dataset, batch_size=64)

class ConvTest(torch.nn.Module):
  def __init__(self):
    super(ConvTest, self).__init__()
    self.conv1 = Conv2d(in_channels = 3, out_channels=6, kernel_size=3, stride=1, padding=0)

  def forward(self, x):
    return self.conv1(x)

class MaxPoolTest(torch.nn.Module):
  def __init__(self):
    super(MaxPoolTest, self).__init__()
    self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

  def forward(self, x):
    x = self.maxpool(x)
    return x

conv_test = ConvTest()
maxpool_test = MaxPoolTest()

writer = SummaryWriter("../logs")

step = 0
for data in dataload:
  imgs, targets = data
  writer.add_images("input", imgs, step)
  
  output1 = conv_test(imgs)
  output1 = torch.reshape(output1, (-1, 3, 30, 30))
  writer.add_images("conv output", output1, step)

  output2 = maxpool_test(imgs)
  writer.add_images("maxpool output", output2, step)

  output3 = maxpool_test(output1)
  writer.add_images("conv_maxpool_output", output3, step)

  step += 1
```
