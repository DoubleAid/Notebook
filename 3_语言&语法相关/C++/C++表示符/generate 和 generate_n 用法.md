# generate
**参数介绍**
+ 容器遍历的起始位置 iterator
+ 容器遍历的终止位置 iterator
+ 特定的动作函数

返回值为`void`

**作用** 以指定动作运算结果填充指定范围内的元素内容

**例子**
```cpp
vector<int> sint(5, 0);
int i = 0;
generate(sint.begin(), sint.end(), [&i](){
    return i++;
});
```

# generate_n
**参数介绍**
+ 容器遍历起始位置
+ 容器遍历的个数
+ 特定的动作函数

返回值 返回一个迭代器
作用 以指定动作的运算结果填充n个元素内容
