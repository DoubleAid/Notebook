# unpack 的 用法

### 参考连接 

+ [次要](https://blog.csdn.net/gracioushe/article/details/5915900)
+ [主要参考](https://cloud.tencent.com/developer/beta/article/1406350)

### 用法介绍

Python 中 按照一定的格式去除某字符串中的子字符串， 使用 `struct.unpack()` 是非常高效的

### 1. 设置 format 的格式， 如下

```python
# 取前5个字符， 跳过四个字符， 再取3个字符
format = '5s 4x 3s'
```

### 2. 使用 struct.unpack 获取字符串

```python
import struct
print(struct.unpack(format, "tast string"))
>>> ("test", "ing")
```

| Format | C Type | Python | Notes |
| --- | --- | --- | --- |
| x | pad byte | no value | 忽略符号 |
| c | char | string length 1 | |
| b | signed char | integer | |
| B | unsigned char | integer | |
| h | short | integer | |
| H | unsigned short | integer | |
| i | int | integer | |
| I | unsigned int | long | |
| l | long | integer | |
| L | unsigned long | long | |
| q | long long | long | |
| Q | unsigned long long | long | |
| f | float | float | |
| d | double | float | |
| s | char[] | string | |
| p | char[] | string | |
| P | void* | interger | |


| Character | Byte Order | Size | Alignment |
| ---- | ---- | ---- | ---- |
| @ | native | native | native |
| = | native | standard | none |
| < | little-endian | standard | none |
| > | big-endian | standard | none |
| ! | network (= big-endian) | standard | none |
