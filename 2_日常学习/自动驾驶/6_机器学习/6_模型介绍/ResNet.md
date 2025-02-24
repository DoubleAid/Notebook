# ResNet

ResNet（Residual Network，残差网络）是一种深度卷积神经网络架构，由微软研究院的 Kaiming He 等人在 2015 年提出，并在 ImageNet 2015 挑战赛中取得了优异的成绩。
ResNet 的核心创新是引入了**残差学习（Residual Learning）**机制，有效解决了深度网络训练中的梯度消失和梯度爆炸问题，使得训练非常深的网络成为可能。

## 1. 背景与动机

随着深度学习的发展，人们发现增加网络的深度可以提高模型的性能，但同时也带来了新的挑战：

+ 梯度消失问题：在反向传播过程中，梯度会随着层数的增加而逐渐变小，导致网络难以训练。
+ 梯度爆炸问题：在某些情况下，梯度可能会变得非常大，导致训练过程不稳定。
+ ResNet 通过引入残差学习机制，解决了这些问题，使得训练非常深的网络成为可能。

## 2. 残差学习的基本思想

残差学习的核心思想是：如果一个较浅的网络能够学习到某个函数 H(x)，那么一个更深的网络可以通过在较浅网络的基础上添加恒等映射（Identity Mapping）来学习到相同的函数。
具体来说，ResNet 引入了一个残差块（Residual Block），通过跳跃连接（Skip Connection）将输入直接传递到后面的层。

## 3. ResNet 的结构

ResNet 的结构由多个残差块组成，每个残差块包含两个或多个卷积层，并通过跳跃连接将输入直接传递到后面的层。

残差块的结构

一个典型的残差块包含以下部分：

+ 两个卷积层：通常包含批量归一化（Batch Normalization）和 ReLU 激活函数。
+ 跳跃连接：将输入直接传递到后面的层，与卷积层的输出相加。

示例：

```cpp
x -> Conv -> BN -> ReLU -> Conv -> BN -> + -> ReLU -> y
                     |__________________________|
```

跳跃连接的作用
跳跃连接允许梯度直接传播到前面的层，从而缓解了梯度消失问题。同时，它也使得网络能够学习到恒等映射，即使在添加更多层时也不会影响网络的性能。

## 4. ResNet 的变体

ResNet 有多个变体，常见的有：

ResNet-18：18 层的 ResNet，包含 18 个卷积层。

ResNet-34：34 层的 ResNet。

ResNet-50：50 层的 ResNet。

ResNet-101：101 层的 ResNet。

ResNet-152：152 层的 ResNet。

## 5. ResNet 的优点

+ 解决梯度消失问题：通过跳跃连接，梯度可以直接传播到前面的层，缓解了梯度消失问题。
+ 提高训练效率：残差学习使得训练非常深的网络成为可能，提高了模型的性能。
+ 灵活性高：ResNet 的结构简单，易于实现和扩展，适用于多种任务和数据集。

## 6. ResNet 的应用场景

ResNet 在计算机视觉领域得到了广泛应用，包括但不限于：

+ 图像分类：在 ImageNet 等数据集上取得了优异的成绩。
+ 目标检测：作为特征提取器，用于 Faster R-CNN 等目标检测算法。
+ 语义分割：在 FCN 等语义分割算法中作为骨干网络。

## 7. 示例代码

以下是一个简单的 ResNet 残差块的实现：

```cpp
#include <torch/torch.h>

class ResidualBlock : public torch::nn::Module {
public:
    ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t stride = 1)
        : conv1(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1)),
          bn1(out_channels),
          conv2(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)),
          bn2(out_channels) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
    }

    # x -> conv -> bn -> relu -> conv -> bn -> + -> relu -> y
    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor identity = x;

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);

        x = conv2->forward(x);
        x = bn2->forward(x);

        x += identity;
        x = torch::relu(x);

        return x;
    }

private:
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
};
```

下面是 使用 pytorch 实现 ResNet-18 的示例代码：

### 7.1 Residual Block 的实现

Residual Block 是 ResNet 的核心组件，它通过跳跃连接（Skip Connection）将输入直接传递到后面的层，从而缓解梯度消失问题。

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out
```

### 7.2. ResNet-18 模型的实现

ResNet-18 是一个具有 18 层的 ResNet 模型，包含 4 个残差块层。

```Python
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)  # strides = 1 + 1 * 0 = 1
        self.layer2 = self._make_layer(128, 2, stride=2) # strides = 2
        self.layer3 = self._make_layer(256, 2, stride=2) # 2
        self.layer4 = self._make_layer(512, 2, stride=2) # 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

### 7.3. 使用 ResNet-18 进行图像分类

以下是一个简单的示例，展示如何使用 ResNet-18 进行图像分类任务。

```Python
import torch
import torchvision
import torchvision.transforms as transforms

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = torchvision.datasets.ImageFolder(root='path_to_train_data', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型
model = ResNet18(num_classes=1000)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

print('Finished Training')
```

## 8. 这个卷积层的 stride 如果不为1的化，out的shape 不是和 identity不一样吗，他们怎么相加

你的观察非常敏锐，确实指出了一个关键问题。在 ResNet 的残差块中，如果卷积层的 stride 不为 1，那么输出的特征图（out）的尺寸会与输入的特征图（identity）的尺寸不一致，这会导致无法直接相加。

### 问题分析

在 ResidualBlock 中，如果 stride 不为 1，conv1 的输出特征图的尺寸会减小。例如，如果输入特征图的尺寸为 H×W，stride 为 2，那么输出特征图的尺寸会变为 2H​×2W​。而 identity 的尺寸仍然是 H×W，因此它们无法直接相加。

### 解决方案

为了解决这个问题，shortcut 也需要对输入特征图进行下采样，以匹配 out 的尺寸。这通常通过一个步长为 stride 的 1x1 卷积层来实现。这样，shortcut 的输出特征图的尺寸会与 out 的尺寸一致，从而可以相加。

### 修正代码

以下是修正后的 ResidualBlock 实现：

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out
```

### 关键点解释

shortcut 的设计：

如果 stride 不为 1 或者输入通道数 in_channels 与输出通道数 out_channels 不一致，shortcut 会包含一个 1x1 卷积层和一个批量归一化层。
1x1 卷积层的作用是调整通道数和下采样特征图，使其尺寸与 out 一致。

conv1 和 conv2 的设计：

conv1 使用步长 stride，可能改变特征图的尺寸。
conv2 使用步长 1，保持特征图的尺寸不变。

相加操作：

out 和 identity 的尺寸一致，可以直接相加。

### 示例验证

以下是一个简单的示例，验证修正后的 ResidualBlock 是否正确处理了尺寸不一致的问题：

```Python
import torch

# 创建一个输入张量
input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入特征图的尺寸为 32x32，通道数为 64

# 创建一个 ResidualBlock 实例
block = ResidualBlock(64, 128, stride=2)  # 输入通道数为 64，输出通道数为 128，步长为 2

# 前向传播
output = block(input_tensor)

print("Input shape:", input_tensor.shape)  # 输入形状
print("Output shape:", output.shape)      # 输出形状

# 输出结果
# Input shape: torch.Size([1, 64, 32, 32])
# Output shape: torch.Size([1, 128, 16, 16])
```

### 总结

问题：如果 stride 不为 1，conv1 的输出特征图的尺寸会减小，与 identity 的尺寸不一致。
解决方案：shortcut 通过 1x1 卷积层和批量归一化层对输入特征图进行下采样，使其尺寸与 out 一致。
验证：通过简单的示例验证了修正后的 ResidualBlock 能够正确处理尺寸不一致的问题。
通过这种方式，ResNet 的残差块可以正确地处理不同尺寸的特征图，从而实现高效的残差学习。
