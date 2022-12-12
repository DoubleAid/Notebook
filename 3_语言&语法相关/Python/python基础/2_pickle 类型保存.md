pickle 的用法主要有两种

## 第一种 直接保存到文件
```python
import pickle

def save(data):
    file_path = "path/to/file"
    f = open(file_path, 'wb')
    pickle.dump(data, f)
    f.close()

def read(file_path):
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()
    return data
```

## 第二种 保存成数据
```python
import pickle

def save(data):
  data_pkl = pickle.dumps(data)
  # 可以将相应的二进制数据转化成 字符串
  # data_str = data_pkl.encode(encoding='latin1')
  # 对应的decode
  # data_pkl = data_str.decode(encoding='latin1')
  return data_pkl

def read(data_pkl)
  data = pickle.loads(data_pkl)
  return data
```

## 问题
我尝试 将 pickle 数据保存到数据库中时， 遇到保存的数据存在乱码， 没法解决， 先暂时保存在临时文件中