# 析构函数=default的意义

在 C++ 中，析构函数 = default 是一种特殊的语法，用于显式地要求编译器生成默认的析构函数。这种用法在 C++11 中引入，主要用于以下几个目的：

## 1. 显式启用默认析构函数

在某些情况下，你可能需要显式地启用默认的析构函数，即使类中已经定义了其他特殊成员函数（如构造函数、拷贝构造函数、移动构造函数等）。
使用 析构函数 = default 可以确保编译器生成默认的析构函数。

示例

```cpp
#include <iostream>

class MyClass {
public:
    MyClass() = default;  // 默认构造函数
    MyClass(const MyClass&) = default;  // 默认拷贝构造函数
    MyClass(MyClass&&) = default;  // 默认移动构造函数
    ~MyClass() = default;  // 默认析构函数
};

int main() {
    MyClass obj;
    return 0;
}
```

## 2. 避免隐式删除

在某些情况下，编译器可能会隐式删除默认的析构函数，例如：

+ 类中包含非静态成员变量，其类型没有默认析构函数。
+ 类中包含继承的析构函数，但继承的析构函数被删除或不可访问。

使用 析构函数 = default 可以显式地要求编译器生成默认的析构函数，避免隐式删除。

示例

```cpp
#include <iostream>

class NonDefaultDestructible {
public:
    ~NonDefaultDestructible() = delete;  // 删除析构函数
};

class MyClass {
    NonDefaultDestructible ndd;
public:
    ~MyClass() = default;  // 显式要求生成默认析构函数
};

int main() {
    MyClass obj;  // 编译器会生成默认析构函数
    return 0;
}
```

## 3. 优化编译器生成的代码

显式使用 析构函数 = default 可以让编译器生成更高效的默认析构函数。这在某些情况下可以提高代码的性能。

## 4. 提高代码的可读性和可维护性

显式声明 析构函数 = default 可以让代码更清晰，明确地告诉读者和编译器，你希望使用默认的析构函数。这有助于提高代码的可读性和可维护性。

## 5. 总结

+ 显式启用默认析构函数：即使类中已经定义了其他特殊成员函数，析构函数 = default 可以确保编译器生成默认的析构函数。
+ 避免隐式删除：在某些情况下，编译器可能会隐式删除默认的析构函数。使用 析构函数 = default 可以避免这种情况。
+ 优化编译器生成的代码：显式声明 析构函数 = default 可以让编译器生成更高效的默认析构函数。
+ 提高代码的可读性和可维护性：显式声明 析构函数 = default 可以让代码更清晰，明确地告诉读者和编译器你的意图。

通过合理使用 析构函数 = default，可以确保类的行为符合预期，同时提高代码的可读性和可维护性。
