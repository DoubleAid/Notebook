
## 单前导下划线

```
_val
```

当涉及到变量和方法名称时，单个下划线前缀有一个约定俗成的含义。 它是对程序员的一个提示 - 意味着Python社区一致认为它应该是什么意思，但程序的行为不受影响。

下划线前缀的含义是告知其他程序员：以单个下划线开头的变量或方法仅供内部使用。 该约定在PEP 8中有定义。

```python
class Test:
   def __init__(self):
       self.foo = 11
       self._bar = 23

t = Test()
print(t.foo)  # 输出 11
print(t._bar)   # 输出 23
```

你会看到_bar中的单个下划线并没有阻止我们“进入”类并访问该变量的值。
这是因为Python中的单个下划线前缀仅仅是一个约定 - 至少相对于变量和方法名而言。

但是，前导下划线的确会影响从模块中导入名称的方式。
假设你在一个名为my_module的模块中有以下代码：

```python
def external_func():
   return 23

def _internal_func():
   return 42
```

当使用通配符 `*` 导入时， 会忽略掉单前导下划线的函数

```python
from module import *
print(external_func())  # 23
print(_internal_func())  # NameError: "name '_internal_func' is not defined"
```

顺便说一下，应该避免通配符导入，因为它们使名称空间中存在哪些名称不清楚。 为了清楚起见，坚持常规导入更好。
与通配符导入不同，常规导入不受前导单个下划线命名约定的影响：

```python
import module
print(module.external_func())  # 23
print(module._internal_func())  # 42
```

## 单末尾下划线

```
val_
```

有时候，一个变量的最合适的名称已经被一个关键字所占用。 因此，像class或def这样的名称不能用作Python中的变量名称。 在这种情况下，你可以附加一个下划线来解决命名冲突：
```
>>> def make_object(name, class):
SyntaxError: "invalid syntax"

>>> def make_object(name, class_):
...    pass
```
总之，单个末尾下划线（后缀）是一个约定，用来避免与Python关键字产生命名冲突。 PEP 8解释了这个约定。

## 双前导下划线

```
__val
```

双下划线前缀会导致Python解释器重写属性名称，以避免子类中的命名冲突。
我知道这听起来很抽象。 因此，我组合了一个小小的代码示例来予以说明：

```python
class Test:
   def __init__(self):
       self.foo = 11
       self._bar = 23
       self.__baz = 23
```

让我们用内置的dir()函数来看看这个对象的属性：

```python
>>> t = Test()
>>> dir(t)
['_Test__baz', '__class__', '__delattr__', '__dict__', '__dir__',
'__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
'__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__',
'__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
'__setattr__', '__sizeof__', '__str__', '__subclasshook__',
'__weakref__', '_bar', 'foo']
```

以上是这个对象属性的列表。 让我们来看看这个列表，并寻找我们的原始变量名称foo，_bar和__baz - 我保证你会注意到一些有趣的变化。

self.foo变量在属性列表中显示为未修改为foo。
self._bar的行为方式相同 - 它以_bar的形式显示在类上。 就像我之前说过的，在这种情况下，前导下划线仅仅是一个约定。 给程序员一个提示而已。
然而，对于self.__baz而言，情况看起来有点不同。 当你在该列表中搜索__baz时，你会看不到有这个名字的变量。
__baz出什么情况了？

如果你仔细观察，你会看到此对象上有一个名为_Test__baz的属性。 这就是Python解释器所做的名称修饰。 它这样做是为了防止变量在子类中被重写。

让我们创建另一个扩展Test类的类，并尝试重写构造函数中添加的现有属性：

```python
class ExtendedTest(Test):
   def __init__(self):
       super().__init__()
       self.foo = 'overridden'
       self._bar = 'overridden'
       self.__baz = 'overridden'
```

现在，你认为foo，_bar和__baz的值会出现在这个ExtendedTest类的实例上吗？ 我们来看一看：

```python
>>> t2 = ExtendedTest()
>>> t2.foo
'overridden'
>>> t2._bar
'overridden'
>>> t2.__baz
AttributeError: "'ExtendedTest' object has no attribute '__baz'"
```

等一下，当我们尝试查看t2 .__ baz的值时，为什么我们会得到AttributeError？ 名称修饰被再次触发了！ 事实证明，这个对象甚至没有__baz属性：

```python
>>> dir(t2)
['_ExtendedTest__baz', '_Test__baz', '__class__', '__delattr__',
'__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
'__getattribute__', '__gt__', '__hash__', '__init__', '__le__',
'__lt__', '__module__', '__ne__', '__new__', '__reduce__',
'__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__',
'__subclasshook__', '__weakref__', '_bar', 'foo', 'get_vars']
```

正如你可以看到__baz变成_ExtendedTest__baz以防止意外修改：

```python
>>> t2._ExtendedTest__baz
'overridden'
```python
但原来的_Test__baz还在：

```
>>> t2._Test__baz
42
```

双下划线名称修饰对程序员是完全透明的。 下面的例子证实了这一点：

```python
class ManglingTest:
   def __init__(self):
       self.__mangled = 'hello'

   def get_mangled(self):
       return self.__mangled

>>> ManglingTest().get_mangled()
'hello'
>>> ManglingTest().__mangled
AttributeError: "'ManglingTest' object has no attribute '__mangled'"
```

名称修饰是否也适用于方法名称？ 是的，也适用。名称修饰会影响在一个类的上下文中，以两个下划线字符（"dunders"）开头的所有名称：

```python
class MangledMethod:
   def __method(self):
       return 42

   def call_it(self):
       return self.__method()

>>> MangledMethod().__method()
AttributeError: "'MangledMethod' object has no attribute '__method'"
>>> MangledMethod().call_it()
42
```

这是另一个也许令人惊讶的运用名称修饰的例子：

```python
_MangledGlobal__mangled = 23

class MangledGlobal:
   def test(self):
       return __mangled

>>> MangledGlobal().test()
23
```

在这个例子中，我声明了一个名为_MangledGlobal__mangled的全局变量。然后我在名为MangledGlobal的类的上下文中访问变量。由于名称修饰，我能够在类的test()方法内，以__mangled来引用_MangledGlobal__mangled全局变量。

Python解释器自动将名称__mangled扩展为_MangledGlobal__mangled，因为它以两个下划线字符开头。这表明名称修饰不是专门与类属性关联的。它适用于在类上下文中使用的两个下划线字符开头的任何名称。

## 双前导和双末尾下划线

```
__val__
```

也许令人惊讶的是，如果一个名字同时以双下划线开始和结束，则不会应用名称修饰。 由双下划线前缀和后缀包围的变量不会被Python解释器修改：

class PrefixPostfixTest:
   def __init__(self):
       self.__bam__ = 42

>>> PrefixPostfixTest().__bam__
42
但是，Python保留了有双前导和双末尾下划线的名称，用于特殊用途。 这样的例子有，__init__对象构造函数，或__call__ --- 它使得一个对象可以被调用。

这些dunder方法通常被称为神奇方法 - 但Python社区中的许多人（包括我自己）都不喜欢这种方法。

最好避免在自己的程序中使用以双下划线（“dunders”）开头和结尾的名称，以避免与将来Python语言的变化产生冲突。

## 单下划线

```
_
```

按照习惯，有时候单个独立下划线是用作一个名字，来表示某个变量是临时的或无关紧要的。

例如，在下面的循环中，我们不需要访问正在运行的索引，我们可以使用“_”来表示它只是一个临时值：

```python
>>> for _ in range(32):
...    print('Hello, World.')
```
