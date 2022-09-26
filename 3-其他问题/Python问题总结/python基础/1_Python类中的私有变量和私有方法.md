在 python 中定义私有变量只需要在变量名或函数名前加上 “__” 两个下划线， 那么这个函数或变量就是私有的了

在内部，python 使用一种 name mangling 技术， 将 __membername 替换成 _classname__membername, 也就是说， 类的内部定义中， 所有的以双下划线开始的名字都被翻译成 类名加成员名的形式

```
私有变量：实例._类名__变量名
私有方法：实例._类名__方法名()
```

其实， Python 并没有真正的私有化支持， 可以用下划线得到伪私有， 要尽量避免定义以下划线开头的变量

(1). _xxx  "单下划线"开始的成员变量叫做保护变量， 意思是只有类实例和子实例能访问到这些变量， 需要通过类提供的借口进行访问； 不能用 “from module import *” 导入
(2). __xxx  类中的私有变量/方法名 “双下划线”开始的是私有成员， 意思是 只有类对象自己能访问， 连子类的对象也不能访问到这个数据
(3). __xxx__  系统定义名字， 前后均有一个“双下划线”， 代表python里特殊方法专用的表示， 如 __init__() 代表类的构造函数，

```python
class A(object):
    def __init__(self):
        self.__data = []  # 翻译成 self._A__data = []
    def print(self):
        print(self.__data)  # 翻译成 self._A__data
    def add(self, item):
        self.__data.append(item)


a = A()
a.add("hello")
a.print()
print(a.__data) # 外界不能访问私有变量， AttributeError： “A” object has 弄attribute ‘__data’
print(a._A__data) # 通过这种方式， 在外面也能够访问 “私有” 变量
```