# transform 的结构和用法

transform 主要是各种转化方法

+ transform 该如何使用
+ 为什么我们需要 Tensor 数据类型

## tensor 数据类型

在 transform 中 的 ToTensor 类

```python
class ToTensor(object):
    """Convert a 'PIL Image' or 'numpy.ndarray' to tensor
    """
    def __call__(self, pic):
        """
        返回 tensor 类型的图片
        """
        return F.to_tensor(pic)
```

```python
from PIL import Image
from torchvision import transform
from torch.utils.tensorboard import SummaryWriter

img_path = "xxx/xxx.jpg"
img = Image.open(img_path)

# 如何使用 transform
tensor_trans = transform.ToTensor()
tensor_img = tensor_trans(img)

writer = SummaryWriter("logs")
writer.add_image("ToTensor", tensor_img)
```

## Compose 方法

用于将几个方法结合在一起使用

```python
transform.Compose([
    transform.CenterCrop(10),
    transform.ToTensor(),
])
```

## ToTensor 方法

用于 tensorboard 的使用
具体使用方法参考 tensor 数据类型

## ToPILImage 方法

把 tensor 数据类型转化成 PIL 数据类型

## Normalize 方法

通过均值和标准差把一张tensor图片标准化, 让数据分散在 [-x, x] 之间

```python
trans_norm = transform.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
```

## Resize 方法

将一个 PIL image 重新计算大小， 初始化有两个输入 ： size 和 interpolation
如果 size 是一个序列， 指定高度和宽度， 会进行相应的转换
如果 size 是一个值， 就会用图像较小的边去匹配这个值

```python
print(img.size)
trans_resize = transform.Resize((512, 512))
img_resize = trans_resize(img) # 返回值是 PIL image
img_resize = transform.ToTensor(img_resize)
# 或者使用 Compose
trans_comp = transform.Compose([
    transform.Resize((512, 512)),
    transform.ToTensor()
])
img_resize = trans_comp(img)
```
