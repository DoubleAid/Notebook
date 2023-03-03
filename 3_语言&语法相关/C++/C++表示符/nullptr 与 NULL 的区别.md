NULL是一个宏定义，它的值是一个空指针常量，由实现来进行定义。C语言中常数0和(void*)0都是空指针常量；C++中常数0是，而(void*)0 不是

因为C语言中任何类型的指针都可以(隐式地)转换为void* 型，反过来也行；而C++中void* 型不能隐式地转换为别的类型指针(例如：
int* p = (void*)0，使用C++编译器编译会报错)。

可以查看到NULL的宏定义内容：NULL在C和C++中的定义不同，C中NULL为（void * )0，而C++中NULL为整数0。

#### nullptr
为了避免上面的情况出现，C++11引入了一个新关键字nullptr(也有的称之为：空指针常量)，它的类型为std::nullptr_t。
在C++中，void * 不能隐式地转化为任意类型的指针(可以强制转化)，但空指针常数可以隐式地转换为任意类型的指针类型。

#### nullptr与nullptr_t
在stddef.h中有如下的描述：
```c++
typedef decltype(nullptr) nullptr_t;
```
nullptr_t是一种数据类型，而nullptr是该类型的一个实例。通常情况下，也可以通过nullptr_t类型创建另一个新的实例；
所有定义为nullptr_t类型的数据都是等价的，行为也是完全一致的；
std::nullptr_t类型，并不是指针类型，但可以隐式转换成任意一个指针类型(注意不能转换为非指针类型，强转也不行)；
nullptr_t类型的数据不适用于算术运算表达式。但可以用于关系运算表达式(仅能与nullptr_t类型数据或指针类型数据进行比较，
当且仅当关系运算符为==、<=、>=等时)。

#### nullptr与NULL的区别
NULL是一个宏定义，C++中通常将其定义为0，编译器一般优先把它当作一个整型常量(C标准下定义为(void*）0)；
nullptr是一个编译期常量，其类型为nullptr_t。它既不是整型类型，也不是指针类型；
在模板推导中，nullptr被推导为nullptr_t类型，仍可隐式转为指针。但0或NULL则会被推导为整型类型；
要避免在整型和指针间进行函数重载。因为NULL会被匹配到整型形参版本的函数，而不是预期的指针版本。

#### nullptr与(void*)0的区别
nullptr到任意类型指针的转换是隐式的(尽管nullptr不是指针类型，但仍可当指针使用)；
(void*)0只是一个强制转换表达式，其返回void*指针类型，只能经过类型转换到其他指针才能用。
例如：

```c++
#include <iostream>

int main(int argc, char *argv[]) {
void* px = NULL;
// int* py = (void*)0;         //编译错误，不能隐式将void*转为int*类型
int* pz = (int*)px;           //void*不能隐式转为int*，必须强制转换！

int* pi = nullptr;            //ok！nullptr可以隐式转为任何其他指针类型
void* pv = nullptr;           //ok! nullptr可以隐式转为任何其他指针类型

	return 0;
}
```

#### 总结
NULL在C语言中是(void * )0，在C++中却是0。这是因为在C++中void * 类型是不允许隐式转换成其他指针类型的，所以之前C++中用0来代表空指针。
但是，在重载整型和指针的情况下，会出现匹配错误的情况。所以，C++11加入了nullptr，可以保证在任何情况下都代表空指针。