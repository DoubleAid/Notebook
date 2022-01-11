## <font color="deepskyblue">np.dot()</font>
如果参与运算的是两个一维函数， 那么得到的结果是两个数组的内积
```python
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
print(np.dot(a, b))
>> 14
```
如果参与运算的是两个二维数组， 那么得到的是矩阵乘积， 两个参与运算的矩阵需要满足矩阵乘法的规则， 但是官方更推荐使用np.matmul() 和 @ 用于矩阵乘法
```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(np.dot(A, B))
>> [
    [22, 28],
    [49, 64]
]
```

## <font color="deepskyblue">np.multiply() 和 *</font>
星号 和 np.multiply()方法 是 针对标量的运算， 放参与运算的是两个数组时，得到的结果是两个数组进行对应位置的乘积， 输出的结果与参与运算的数组或者矩阵的大小一致
```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
B = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
a = np.array([1, 2, 3])
b = np.array([1 ,2, 3])
print(np.multiply(A, B))
print(np.multiply(a, b))
>> [
    [1, 4, 9],
    [16, 25, 36]
]
>> [1 4 9]
```

## <font color="deepskyblue">np.matmul() 和 @</font>
matmul 是 matrix multiply的缩写， 是专门用于矩阵乘法的函数。 另外，@运算方法和matmul()是一样的作用，只是便与书写
```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
B = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(np.matmul(A, B))
print(A @ B)
>> [[22 28]
    [49 64]]
   [[22 28]
    [49 64]]
```