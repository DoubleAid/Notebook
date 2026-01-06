# Bottleneck 瓶颈的核心理念

可以先观察以下下面这个自定义的模型

```python
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1,
                 downsample=False, upsample=False, stride=1):
        super(Bottleneck, self).__init__()
        inter_channels = out_channels // 4
        
        self.downsample = downsample
        self.upsample = upsample
        self.stride = stride
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        
        if downsample:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, 
                                  stride=2, padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3,
                                  stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        
        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 下采样或上采样层
        if downsample:
            self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        elif upsample:
            self.upsample_layer = nn.ConvTranspose2d(out_channels, out_channels, 
                                                     kernel_size=2, stride=2)
        else:
            self.maxpool = None
            self.upsample_layer = None
        
        # 快捷连接
        if in_channels != out_channels or downsample or (stride != 1 and not upsample):
            if downsample:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, indices=None, output_size=None):
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 快捷连接
        if self.downsample:
            identity, idx = self.maxpool(identity)
        identity = self.shortcut(identity)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        if self.upsample and self.upsample_layer is not None:
            out = self.upsample_layer(out)
        
        if self.downsample:
            return out, idx
        else:
            return out
```

首先他会缩小整体的通道数，随后再扩展通道数，为什么这么做呢

**Bottleneck 设计是为了减少计算量和参数数量，同时保持甚至提升网络性能。**

## 计算对比

假设一个输入64通道，输出128通道的普通计算

### 1. 情况A：不使用 Bottleneck（普通残差块）

输入：64通道 → 128通道

直接使用 3x3 的卷积，每一次的卷积核为 64x3x3 = 576
需要128通道，所以 共需要优化的参数为 576x128 = 73728

### 2. 情况B: 使用 Bottleneck

输入：64通道 → 中间：32通道 → 输出：128通道

+ 首先是 1x1卷积降为 32 通道：需要的参数为 32x64x1x1 = 2048
+ 接着进行一次 3x3 卷积：需要的参数为 32x32x3x3 = 9216
+ 最后再扩大通道数：需要的参数为 32x128x1x1 = 4096

总计需要的参数为 2048 + 9216 + 4096 = 15360

相比较可以看出来，两者所需要的参数大幅降低了

## 信息流动分析

```
输入 (64) → 1×1 conv → 中间 (32) → 3×3 conv → 中间 (32) → 1×1 conv → 输出 (128)
     ↓                                        ↑
      - - - - - - 残差连接 - - - - - - - - - -
```

虽然中间通道数减少，但：

+ 1×1 卷积：先降维，减少后续 3×3 卷积的计算量
+ 3×3 卷积：在低维度空间提取空间特征
+ 1×1 卷积：再升维到目标通道数
+ 残差连接：确保信息不丢失

### 为什么有效？

+ 计算效率：3×3 卷积是计算量最大的部分，在低维度（32通道）进行，效率高
+ 非线性增强：增加了两个额外的 ReLU 激活函数
+ 特征重组：1×1 卷积能进行通道间的特征重组和组合

## 总结

+ 经典 Bottleneck​ 是为了效率，不是信息最大化
+ 在编码器中，可以适当调整中间通道数
+ 如果计算资源充足，可以使用扩展而不是压缩的设计
+ 关键是平衡效率、参数量和特征提取能力