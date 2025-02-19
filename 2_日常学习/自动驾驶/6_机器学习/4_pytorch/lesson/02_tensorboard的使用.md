原本是tensorflow的可视化工具，pytorch从1.2.0开始支持tensorboard。
使用 `SummaryWriter` 写入 到 `log_dir`, tensorboard 可以解析相应目录下的文件
## 使用举例
```python3
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")

# 主要使用的两个方法
# 添加图片
writer.add_image()
# 添加标量（即图表值）
writer.add_scalar()
```

## 打开 tensorboard
```
tensorboard --logdir=logs--port=6007
```
通过 `localhost:6007` 访问
## add_scalar
```
def add_scalar(self, tag, scalar_value, global_step=None, walltime=None)
```
+ tag (string): 作为图表的标识符，也就是图表的标题，用于指定标量数据属于哪一个图表
+ scalar_value (string or float): 需要保存的值
+ global_step (int): x轴值

以 `y=2x`为例
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")
for i in range(100):
  writer.add_scalar("y=2x", 2*i, i)
writer.close()
```
## add_image
```
def add_image(self, img_tensor, global_step=None, walltime=None, dataformats="CHW")
```
+ tag (string): 图片的标题
+ img_tensor (torch.tensor, numpy.array or string): 图片数据

注意图片的形状 默认是 `(3, H, W)`, 如果使用其他形式的格式， 比如 `(H, W, 3)` 或者 `(1, H, W)` 或者 `(H, W)`, 需要修改 `dataformats` 的格式 "CHW", "HWC", "HW"

```python
import numpy
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")
image_path = "./image.jpg"
# 首先使用 opencv 读取图片
img = Image.open(image_path)
image_array = np.array(img)
writer.add_image("test", image_array, 1, dataformats="HWC")
writer.close()
```