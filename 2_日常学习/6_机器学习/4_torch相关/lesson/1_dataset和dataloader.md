# Dataset
`class torch.utils.data.Dataset` 表示 Dataset 的抽象类。
所有其他数据集都应该进行子类化。 所有子类应该 重写 `__len__` 和 `__getitem__`, 前者提供了数据集的大小， 后者支持整数索引
 