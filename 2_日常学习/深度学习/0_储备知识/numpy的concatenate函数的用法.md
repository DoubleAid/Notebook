### concatenate 函数用于数组的拼接
函数形式 `concatenate((a1, a2, a3, ...), axis=0)`

#### 参数
+ 传入的第一个参数必须是一个多个数组的元祖或者列表
+ axis 表示拼接的方向， 默认 axis=0， 沿纵向拼接，axis=1表示沿横向拼接

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

np.concatenate((a, b), axis=0)
>>> array([
    [1, 2],
    [3, 4],
    [5, 6]
])

np.concatenate((a, b), axis=1)
>>> array([
    [1, 2, 5],
    [3, 4, 6]
])
```