官方文档 https://pytorch.org/docs/stable/nn.html

nn 是 Nerual Network 的缩写

torch.nn 是 torch.nn.functional 的进一步封装

+ Container 用于定义神经网络的骨架
下面是向骨架中添加的要素
+ Convolution Layers
+ Pooling Layers

# Container
+ Module          所有神经网络模型的基本类
+ Sequential 
+ ModuleList
+ ModuleDict
+ ParameterList
+ ParameterDict

继承 Module 的子类主要实现两个方法， 初始化 `__init__(self)` 和 `forward(self, x)`
```
class Model(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

