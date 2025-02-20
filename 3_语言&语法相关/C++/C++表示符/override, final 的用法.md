# override 和 final 的用法

## override 的用法

override 是 C++11 引入的一个关键字，用于显式标记类中的虚函数（virtual function）是覆盖（override）了基类中的虚函数。
使用 override 可以提高代码的可读性和安全性，同时帮助编译器检测错误。

### 1. override 的基本用法

#### 1.1 声明覆盖基类的虚函数

在派生类中，使用 override 关键字可以显式声明一个虚函数覆盖了基类中的虚函数。

示例：

```cpp
#include <iostream>

class Base {
public:
    virtual void print() const {
        std::cout << "Base::print" << std::endl;
    }
};

class Derived : public Base {
public:
    void print() const override {  // 使用 override 显式声明覆盖
        std::cout << "Derived::print" << std::endl;
    }
};

int main() {
    Derived d;
    d.print();  // 输出：Derived::print
    return 0;
}
```

### 2. override 的优点

#### 2.1 提高代码可读性

使用 override 可以明确地告诉读者，这个函数是覆盖了基类中的虚函数。这使得代码更易于理解和维护。

#### 2.2 编译器检查

如果派生类中的函数没有正确覆盖基类中的虚函数，编译器会报错。这可以避免因拼写错误或签名不匹配而导致的隐藏错误。

示例：

```cpp
class Base {
public:
    virtual void print(int x) const {
        std::cout << "Base::print(" << x << ")" << std::endl;
    }
};

class Derived : public Base {
public:
    void print() const override {  // 错误：签名不匹配
        std::cout << "Derived::print" << std::endl;
    }
};
```

编译器错误：
error: 'void Derived::print() const' marked 'override' but does not override

### 3. override 的高级用法

#### 3.1 与 final 结合使用

override 可以与 final 关键字结合使用，表示该虚函数不仅覆盖了基类中的虚函数，而且不允许进一步派生类覆盖它。

示例：

```cpp
class Base {
public:
    virtual void print() const {
        std::cout << "Base::print" << std::endl;
    }
};

class Derived : public Base {
public:
    void print() const override final {  // 使用 override 和 final
        std::cout << "Derived::print" << std::endl;
    }
};

class FurtherDerived : public Derived {
public:
    void print() const override {  // 错误：Derived::print 是 final 的
        std::cout << "FurtherDerived::print" << std::endl;
    }
};

// 编译器错误：
// error: 'void FurtherDerived::print() const' marked 'override', but does not override
```

### 4. 示例代码

#### 4.1 使用 override

```cpp
#include <iostream>

class Base {
public:
    virtual void print() const {
        std::cout << "Base::print" << std::endl;
    }
};

class Derived : public Base {
public:
    void print() const override {
        std::cout << "Derived::print" << std::endl;
    }
};

int main() {
    Derived d;
    d.print();  // 输出：Derived::print
    return 0;
}
```

#### 4.2 使用 override 和 final

```cpp
#include <iostream>

class Base {
public:
    virtual void print() const {
        std::cout << "Base::print" << std::endl;
    }
};

class Derived : public Base {
public:
    void print() const override final {
        std::cout << "Derived::print" << std::endl;
    }
};

int main() {
    Derived d;
    d.print();  // 输出：Derived::print
    return 0;
}
```

### 5. 总结

override 是一个非常有用的工具，用于显式标记类中的虚函数覆盖了基类中的虚函数。它不仅可以提高代码的可读性，还可以帮助编译器检测错误。以下是 override 的主要用法：

+ 声明覆盖基类的虚函数：使用 override 显式声明覆盖。
+ 提高代码可读性：明确告诉读者，这个函数覆盖了基类中的虚函数。
+ 编译器检查：如果派生类中的函数没有正确覆盖基类中的虚函数，编译器会报错。
+ 与 final 结合使用：表示该虚函数不仅覆盖了基类中的虚函数，而且不允许进一步派生类覆盖它。

通过合理使用 override，可以提高代码的可读性和安全性，同时避免因隐藏错误而导致的问题。

## final 的用法

final 是 C++11 引入的一个关键字，用于显式地声明类或虚函数不能被进一步派生或覆盖。final 的使用可以提高代码的可读性和安全性，同时帮助编译器检测错误。

以下是 final 的主要用法和示例。

### 1. final 的基本用法

#### 1.1 禁止虚函数被覆盖

final 可以用于虚函数，表示该虚函数不能被派生类进一步覆盖。

示例：

```cpp
#include <iostream>

class Base {
public:
    virtual void print() const {
        std::cout << "Base::print" << std::endl;
    }
};

class Derived : public Base {
public:
    void print() const override final {  // 使用 final 禁止进一步覆盖
        std::cout << "Derived::print" << std::endl;
    }
};

class FurtherDerived : public Derived {
public:
    void print() const override {  // 错误：Derived::print 是 final 的
        std::cout << "FurtherDerived::print" << std::endl;
    }
};

int main() {
    Derived d;
    d.print();  // 输出：Derived::print

    FurtherDerived fd;
    fd.print();  // 编译器错误
    return 0;
}

// 编译器错误：
// error: 'void FurtherDerived::print() const' marked 'override', but does not override
```

#### 1.2 禁止类被派生

final 也可以用于类，表示该类不能被进一步派生。

示例：

```cpp
#include <iostream>

class Base {
public:
    virtual void print() const {
        std::cout << "Base::print" << std::endl;
    }
};

class FinalClass final : public Base {  // 使用 final 禁止进一步派生
public:
    void print() const override {
        std::cout << "FinalClass::print" << std::endl;
    }
};

class FurtherDerived : public FinalClass {  // 错误：FinalClass 是 final 的
public:
    void print() const override {
        std::cout << "FurtherDerived::print" << std::endl;
    }
};

int main() {
    FinalClass fc;
    fc.print();  // 输出：FinalClass::print

    FurtherDerived fd;  // 编译器错误
    return 0;
}

// 编译器错误：
// error: cannot derive from 'final' base 'FinalClass'
```

### 2. final 的优点

#### 2.1 提高代码可读性

使用 final 可以明确地告诉读者，某个虚函数或类是最终的，不能被进一步覆盖或派生。这使得代码更易于理解和维护。

#### 2.2 编译器检查

如果尝试覆盖 final 声明的虚函数或派生 final 声明的类，编译器会报错。这可以避免因隐藏错误而导致的问题。

### 3. 示例代码

#### 3.1 禁止虚函数被覆盖

```cpp
#include <iostream>

class Base {
public:
    virtual void print() const {
        std::cout << "Base::print" << std::endl;
    }
};

class Derived : public Base {
public:
    void print() const override final {  // 使用 final 禁止进一步覆盖
        std::cout << "Derived::print" << std::endl;
    }
};

int main() {
    Derived d;
    d.print();  // 输出：Derived::print
    return 0;
}
```

#### 3.2 禁止类被派生

```cpp
#include <iostream>

class Base {
public:
    virtual void print() const {
        std::cout << "Base::print" << std::endl;
    }
};

class FinalClass final : public Base {  // 使用 final 禁止进一步派生
public:
    void print() const override {
        std::cout << "FinalClass::print" << std::endl;
    }
};

int main() {
    FinalClass fc;
    fc.print();  // 输出：FinalClass::print
    return 0;
}
```

### 4. 总结

final 是一个非常有用的工具，用于显式声明类或虚函数不能被进一步派生或覆盖。它不仅可以提高代码的可读性，还可以帮助编译器检测错误。以下是 final 的主要用法：

+ 禁止虚函数被覆盖：使用 final 声明虚函数，表示该虚函数不能被派生类进一步覆盖。
+ 禁止类被派生：使用 final 声明类，表示该类不能被进一步派生。
+ 提高代码可读性：明确告诉读者，某个虚函数或类是最终的，不能被进一步修改。
+ 编译器检查：如果尝试覆盖 final 声明的虚函数或派生 final 声明的类，编译器会报错。

通过合理使用 final，可以提高代码的可读性和安全性，同时避免因隐藏错误而导致的问题。
