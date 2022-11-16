dataset 主要用于提供获取数据的方式
重载这个类的时候需要实现两个方法
+ `__getitem__(self, idx)` 获取单个数据
+ `__len__(self)` 获取数据集长度

```python
from torch.utils.data import Dataset
import os
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.label_dir = label_dir
        self.root_dir = root_dir
        self.img_path = os.
```