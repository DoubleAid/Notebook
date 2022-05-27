在深度学习的实现中， 经常出现数组和矩阵的计算， numpy 的数组类（numpy.array）中提供了许多便捷的方法。

- [生成Numpy数组](#生成numpy数组)
- [Numpy的算数运算](#numpy的算数运算)
- [Numpy的N维数组](#numpy的n维数组)

### 生成Numpy数组
```
>>> x = np.array([1, 2, 3])
>>> print(x)
[1 2 3]
>>> type(x)
<class 'numpy.ndarray'>
```

### Numpy的算数运算

```python
>>> x = np.array([1.0, 2.0, 3.0])
>>> y = np.array([2.0, 4.0, 6.0])
>>> x + y
array([3., 6., 9.])
>>> x - y
array([-1., -2., -3.])
>>> x * y
array([ 2.,  8., 18.])
>>> x / y
array([0.5, 0.5, 0.5])
```
这种元素对元素的操作称为 element-wise, 比如 对应元素的乘法 称为 “element-wise product”， 对应元素的操作需要两个数组的 元素数量是相同的

numpy 不仅可以进行 element-wise 运算， 也可以和单一标量组合起来进行运算， 此时是 numpy数组的各个元素和标量之间的运算， 这种功能也被称为广播

```
>>> y
array([2., 4., 6.])
>>> y/2.0
array([1., 2., 3.])
>>> y-1
array([1., 3., 5.])
```

### Numpy的N维数组
Numpy不仅可以生成 一维数组，也可以生成多维数组

```
>>> A = np.array([[1, 2], [3, 4]])
>>> print(A)
[[1 2]
 [3 4]]
>>> A.shape
(2, 2)
>>> A.dtype
dtype('int64')
```
矩阵 A 的形状可以通过shape查看， 矩阵元素的数据类型可以通过 dtype 查看

```
>>> A
array([[1, 2],
       [3, 4]])
>>> B
array([[3, 0],
       [0, 6]])
>>> A + B
array([[ 4,  2],
       [ 3, 10]])
>>> A * B
array([[ 3,  0],
       [ 0, 24]])
```
注意现在仍然是 element-wise， 也就是对应元素的运算

也可以通过标量（单一数值）对矩阵进行算术运算， 这也是基于广播的功能

```

```