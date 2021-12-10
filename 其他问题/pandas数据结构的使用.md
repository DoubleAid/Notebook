Pandas 数据结构 - DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。
```
pandas.DataFrame( data, index, columns, dtype, copy)
```
+ data: 一组数据(ndarray、series, map, lists, dict 等类型)。
+ index: 行标签
+ columns: 列标签
+ dtype： 数据类型（默认即可）
+ copy： 拷贝数据， 默认False

### 例子
```
import pandas as pd

data = {'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}

df = pd.DataFrame(data)

print (df)
```
输出的结果为
```
    site    age
0   Google  10.0
1   Runoob  12.0
2   Wiki    13.0
```

### 参考链接
+ https://www.runoob.com/pandas/pandas-dataframe.html
+ 
