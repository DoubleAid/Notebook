# const, extern, volatile, static 的用法

## const 的用法

const 是 C++ 中的一个关键字，用于声明变量或函数参数为“只读”，即它们的值在声明后不能被修改。
const 的使用可以提高代码的安全性和可读性，同时帮助编译器进行优化。以下是 const 的常见用法和一些最佳实践。

### 1. 基本用法

#### 1.1 声明常量

const 可以用于声明常量，这些常量在声明后不能被修改。
示例：

```cpp
const int MAX_SIZE = 100;  // 常量，值不能被修改
const double PI = 3.14159;
```

#### 1.2 声明指针

const 可以用于声明指针的指向对象为常量，或者指针本身为常量。
示例：

```cpp
const int* ptr = &MAX_SIZE;  // 指针指向的值不能被修改
int* const ptr2 = &MAX_SIZE; // 指针本身不能被修改
const int* const ptr3 = &MAX_SIZE; // 指针和指向的值都不能被修改
```

#### 1.3 声明引用

const 可以用于声明引用为常量引用，即引用的值不能被修改。
示例：

```cpp
const int& ref = MAX_SIZE;  // 常量引用，引用的值不能被修改
```

### 2. 在函数中的使用

#### 2.1 声明函数参数为常量

const 可以用于声明函数参数为常量，确保函数不会修改参数的值。
示例：

```cpp
void print(const std::string& str) {
    std::cout << str << std::endl;
}
```

#### 2.2 声明函数返回值为常量

const 可以用于声明函数返回值为常量，确保返回值不能被修改。
示例：

```cpp
const std::string getGreeting() {
    return "Hello, World!";
}
```

#### 2.3 声明成员函数为常量

const 可以用于声明成员函数为常量，确保成员函数不会修改对象的成员变量。
示例：

```cpp
class MyClass {
public:
    int getValue() const { return value_; }  // 常量成员函数
private:
    int value_;
};
```

### 3. 在类中的使用

#### 3.1 声明成员变量为常量

const 可以用于声明类的成员变量为常量，这些变量在构造函数中初始化后不能被修改。
示例：

```cpp
class MyClass {
public:
    MyClass(int v) : value_(v) {}
    int getValue() const { return value_; }
private:
    const int value_;  // 常量成员变量
};
```

#### 3.2 声明静态成员变量为常量

const 可以用于声明静态成员变量为常量，这些变量在类的作用域内是只读的。
示例：

```cpp
class MyClass {
public:
    static const int MAX_SIZE = 100;  // 常量静态成员变量
};
```

### 4. const 的最佳实践

#### 4.1 使用 const 修饰函数参数

如果函数不需要修改参数的值，应使用 const 修饰参数，以提高代码的安全性和可读性。
示例：

```cpp
void print(const std::string& str) {
    std::cout << str << std::endl;
}
```

#### 4.2 使用 const 修饰返回值

如果函数返回值不需要被修改，应使用 const 修饰返回值。
示例：

```cpp
const std::string getGreeting() {
    return "Hello, World!";
}
```

#### 4.3 使用 const 修饰成员函数

如果成员函数不需要修改对象的状态，应将其声明为 const，以确保函数不会修改对象的成员变量。
示例：

```cpp
class MyClass {
public:
    int getValue() const { return value_; }
private:
    int value_;
};
```

#### 4.4 使用 const 修饰成员变量

如果成员变量在初始化后不需要被修改，应将其声明为 const，以确保变量的值不会被意外修改。
示例：

```cpp
class MyClass {
public:
    MyClass(int v) : value_(v) {}
    int getValue() const { return value_; }
private:
    const int value_;
};
```

### 5. 注意事项

#### 5.1 const 和引用

如果函数参数是引用类型，应使用 const 修饰引用，以确保引用的值不会被修改。
示例：

```cpp
void print(const std::string& str) {
    std::cout << str << std::endl;
}
```

#### 5.2 const 和指针

如果函数参数是指针类型，应使用 const 修饰指针指向的值，以确保指针指向的值不会被修改。
示例：

```cpp
void print(const char* str) {
    std::cout << str << std::endl;
}
```

#### 5.3 const 和 const_cast

如果需要修改 const 修饰的变量，可以使用 const_cast，但应谨慎使用，以避免未定义行为。
示例：

```cpp
const int a = 10;
int* ptr = const_cast<int*>(&a);
*ptr = 20;  // 修改 a 的值，可能导致未定义行为
```

### 6. 总结

const 是 C++ 中一个非常重要的关键字，用于声明变量、函数参数、返回值和成员函数为只读。通过合理使用 const，可以提高代码的安全性和可读性，同时帮助编译器进行优化。以下是 const 的一些主要用法和最佳实践：

+ 声明常量：使用 const 声明常量，确保变量的值不会被修改。
+ 修饰函数参数：使用 const 修饰函数参数，确保函数不会修改参数的值。
+ 修饰返回值：使用 const 修饰函数返回值，确保返回值不会被修改。
+ 修饰成员函数：使用 const 修饰成员函数，确保成员函数不会修改对象的状态。
+ 修饰成员变量：使用 const 修饰成员变量，确保变量的值在初始化后不会被修改。

通过合理使用 const，可以使代码更加健壮、安全和易于维护。

### 7. 在函数后加const的意义

我们定义的类的成员函数中，常常有一些成员函数不改变类的数据成员， 也就是说， 这些函数是 “只读” 函数，而一些函数要修改类数据成员的值。如果把不改变数据成员的函数都加上const关键字进行标识，可以提高程序的可读性和可靠性， 已定义成const的成员函数， 一旦企图修改数据成员的值，编译器就会报错

+ 非静态成员函数后面加入const （加到非成员函数或静态成员后面会产生编译错误）
+ 表示成员函数隐含传入的 this 指针为const指针，决定了在该成员函数中， 任意修改它所在的类的成员的操作都是不允许的，
+ 唯一的例外是对于mutable修饰的成员
+ 加了const的成员函数可以被非const对像和const对象调用
  但不加const的成员函数只能被非const对象调用

C++ 函数前面和后面使用const的作用

+ 前面使用const表示返回值为const
+ 后面加 const 表示函数不可以修改class成员变量

## extern 的用法

extern 是 C++ 中的一个关键字，用于声明变量或函数的链接属性。它告诉编译器，某个变量或函数的定义在其他地方，而不是在当前文件中。extern 的主要用途包括：
在多个源文件中共享全局变量。
声明外部函数的原型。

### 1. extern 的基本用法

#### 1.1 全局变量的共享

当你需要在多个源文件中共享全局变量时，可以使用 extern。
示例：
假设你有两个源文件 file1.cpp 和 file2.cpp，并且需要在它们之间共享一个全局变量 int globalVar。
file1.cpp:

```cpp
#include <iostream>

int globalVar = 10;  // 定义全局变量

void printGlobalVar() {
    std::cout << "GlobalVar in file1: " << globalVar << std::endl;
}
```

file2.cpp:

```cpp
#include <iostream>

extern int globalVar;  // 声明全局变量（定义在其他地方）

void modifyGlobalVar() {
    globalVar = 20;  // 修改全局变量
    std::cout << "GlobalVar in file2: " << globalVar << std::endl;
}
```

main.cpp:

```cpp
#include <iostream>
#include "file1.cpp"  // 假设 file1.cpp 和 file2.cpp 被包含在项目中

int main() {
    printGlobalVar();  // 输出：GlobalVar in file1: 10
    modifyGlobalVar(); // 输出：GlobalVar in file2: 20
    printGlobalVar();  // 输出：GlobalVar in file1: 20
    return 0;
}
```

1.2 外部函数的声明
当你需要在多个源文件中使用同一个函数时，可以在头文件中使用 extern 声明函数原型。
示例：
假设你有两个源文件 file1.cpp 和 file2.cpp，并且需要在它们之间共享一个函数 void sharedFunction()。
shared.h:

```cpp
#ifndef SHARED_H
#define SHARED_H

extern void sharedFunction();  // 声明外部函数

#endif
```

file1.cpp:

```cpp
#include "shared.h"
#include <iostream>

void sharedFunction() {  // 定义外部函数
    std::cout << "Shared function called" << std::endl;
}
```

file2.cpp:

```cpp
#include "shared.h"
#include <iostream>

void callSharedFunction() {
    sharedFunction();  // 调用外部函数
}
```

main.cpp:

```cpp
#include "shared.h"
#include "file1.cpp"  // 假设 file1.cpp 和 file2.cpp 被包含在项目中

int main() {
    sharedFunction();  // 调用共享函数
    callSharedFunction();  // 调用 file2 中的函数
    return 0;
}
```

### 2. extern 的注意事项

#### 2.1 避免重复定义

在一个源文件中定义全局变量或函数时，不要在其他源文件中重复定义。
使用 extern 声明时，确保变量或函数的定义在某个地方存在。

#### 2.2 链接问题

如果声明了 extern 变量或函数，但没有提供定义，链接器会报错。
确保在某个源文件中定义了变量或函数。

#### 2.3 头文件中的声明

通常在头文件中使用 extern 声明全局变量或函数原型。
在源文件中定义这些变量或函数。

### 3. 总结

extern 是一个非常有用的工具，用于在多个源文件中共享全局变量和函数。它的主要用途包括：
共享全局变量：在多个源文件中使用同一个全局变量。
声明外部函数：在头文件中声明函数原型，确保函数在多个源文件中可见。
通过合理使用 extern，可以提高代码的模块化和可维护性。

## volatile 的用法

volatile 是 C++ 中的一个关键字，用于修饰变量，告诉编译器该变量的值可能会在程序运行时被外部因素（如硬件设备、中断服务例程等）修改。
因此，编译器在优化代码时，不能对 volatile 修饰的变量进行某些优化，以确保每次访问该变量时都能读取其最新的值。

### 1. volatile 的主要用途

硬件寄存器访问：

用于访问硬件寄存器，这些寄存器的值可能会被硬件设备修改。

示例：访问内存映射的硬件寄存器。

多线程环境：

用于修饰可能被多个线程访问的变量，确保每次读取的值是最新的。

示例：在多线程程序中，某些变量可能被多个线程读写。

中断服务例程：

用于修饰可能被中断服务例程修改的变量，确保每次访问时都能读取最新的值。

示例：在嵌入式系统中，某些变量可能被中断服务例程修改。

### 2. volatile 的基本用法

#### 2.1 修饰变量

volatile 可以用于修饰变量，确保每次访问该变量时都能读取其最新的值。
示例：

```cpp
volatile int flag = 0;  // 修饰变量
```

#### 2.2 修饰指针

volatile 可以用于修饰指针指向的值，确保每次通过指针访问时都能读取最新的值。
示例：

```cpp复制
volatile int* ptr = nullptr;  // 修饰指针指向的值
```

### 3. 示例代码

#### 3.1 硬件寄存器访问

假设有一个硬件寄存器映射到内存地址 0x1000，你可以使用 volatile 来访问它。

```cpp复制
#include <iostream>

volatile int* hardware_register = reinterpret_cast<volatile int*>(0x1000);

void read_register() {
    int value = *hardware_register;  // 读取硬件寄存器的值
    std::cout << "Register value: " << value << std::endl;
}

void write_register(int value) {
    *hardware_register = value;  // 写入硬件寄存器
}

int main() {
    write_register(42);
    read_register();
    return 0;
}
```

#### 3.2 多线程环境

在多线程程序中，某些变量可能被多个线程读写，使用 volatile 确保每次读取的值是最新的。

```cpp
#include <iostream>
#include <thread>
#include <atomic>

volatile bool flag = false;

void set_flag() {
    flag = true;
}

void check_flag() {
    while (!flag) {
        // 等待 flag 被设置
    }
    std::cout << "Flag is set!" << std::endl;
}

int main() {
    std::thread t1(check_flag);
    std::thread t2(set_flag);

    t1.join();
    t2.join();
    return 0;
}
```

### 4. volatile 的注意事项

不保证线程安全：

volatile 不提供线程同步机制，不能保证多线程环境下的线程安全。
在多线程环境中，建议使用 std::atomic 或互斥锁（std::mutex）来实现线程同步。

编译器优化：
volatile 会阻止编译器对变量的某些优化，但不会阻止所有优化。
对于复杂的同步需求，volatile 可能不足以保证程序的正确性。

硬件寄存器访问：
在嵌入式系统中，volatile 是访问硬件寄存器的常用方式，但需要确保寄存器地址的正确性。

### 5. 总结

volatile 是一个非常有用的工具，用于修饰可能被外部因素修改的变量。它的主要用途包括：

+ 硬件寄存器访问：确保每次访问硬件寄存器时都能读取最新的值。
+ 多线程环境：确保每次读取的值是最新的，但不提供线程同步机制。
+ 中断服务例程：确保每次访问时都能读取最新的值。

通过合理使用 volatile，可以提高代码的正确性和可维护性，但需要注意它不提供线程同步机制。在多线程环境中，建议使用 std::atomic 或互斥锁来实现线程同步。

## static 的用法

static 是 C++ 中的一个关键字，具有多种用途，包括控制变量的存储期、链接属性以及类成员的特性。以下是 static 的主要用法和详细解释。

### 1. static 的基本用法

#### 1.1 局部静态变量

在函数内部声明的 static 变量具有静态存储期，即它们在程序的整个运行期间都存在，但只在第一次执行到该变量的声明时初始化。
示例：

```cpp
#include <iostream>

void printCount() {
    static int count = 0;  // 局部静态变量
    count++;
    std::cout << "Count: " << count << std::endl;
}

int main() {
    printCount();  // 输出：Count: 1
    printCount();  // 输出：Count: 2
    return 0;
}
```

#### 1.2 全局静态变量

在文件作用域中声明的 static 变量具有内部链接属性，即它们只能在声明它们的文件中访问，不能被其他文件访问。
示例：

```cpp
// file1.cpp
#include <iostream>

static int globalVar = 10;  // 全局静态变量

void printGlobalVar() {
    std::cout << "GlobalVar in file1: " << globalVar << std::endl;
}
```

```cpp
// file2.cpp
#include <iostream>

// 外部声明（尝试访问 file1.cpp 中的 globalVar）
// extern int globalVar;  // 错误：globalVar 是静态的，无法从其他文件访问

void modifyGlobalVar() {
    // globalVar = 20;  // 错误：globalVar 是静态的，无法从其他文件访问
}
```

#### 1.3 静态成员变量

在类中声明的 static 成员变量属于类本身，而不是类的某个对象。静态成员变量需要在类外进行定义和初始化。
示例：

```cpp
#include <iostream>

class MyClass {
public:
    static int count;  // 静态成员变量
};

int MyClass::count = 0;  // 静态成员变量的定义和初始化

void incrementCount() {
    MyClass::count++;
}

int main() {
    incrementCount();
    std::cout << "Count: " << MyClass::count << std::endl;  // 输出：Count: 1
    return 0;
}
```

#### 1.4 静态成员函数

在类中声明的 static 成员函数属于类本身，而不是类的某个对象。静态成员函数可以访问类的静态成员变量，但不能访问非静态成员变量。
示例：

```cpp
#include <iostream>

class MyClass {
public:
    static int count;  // 静态成员变量

    static void incrementCount() {  // 静态成员函数
        count++;
    }
};

int MyClass::count = 0;  // 静态成员变量的定义和初始化

int main() {
    MyClass::incrementCount();
    std::cout << "Count: " << MyClass::count << std::endl;  // 输出：Count: 1
    return 0;
}
```

### 2. static 的注意事项

#### 2.1 局部静态变量的初始化

局部静态变量在第一次执行到声明时初始化，并且只初始化一次。

#### 2.2 全局静态变量的链接属性

全局静态变量具有内部链接属性，只能在声明它们的文件中访问。

#### 2.3 静态成员变量的定义和初始化

静态成员变量需要在类外进行定义和初始化，即使它们在类内声明为 static。

#### 2.4 静态成员函数的调用

静态成员函数可以通过类名调用，也可以通过对象调用，但它们不能访问非静态成员变量。

### 3. 总结

static 是一个多功能的关键字，用于控制变量的存储期、链接属性以及类成员的特性。以下是 static 的主要用法：

+ 局部静态变量：在函数内部声明的 static 变量具有静态存储期，只在第一次执行到声明时初始化。
+ 全局静态变量：在文件作用域中声明的 static 变量具有内部链接属性，只能在声明它们的文件中访问。
+ 静态成员变量：在类中声明的 static 成员变量属于类本身，需要在类外进行定义和初始化。
+ 静态成员函数：在类中声明的 static 成员函数属于类本身，可以访问类的静态成员变量，但不能访问非静态成员变量。

通过合理使用 static，可以提高代码的模块化和可维护性。
