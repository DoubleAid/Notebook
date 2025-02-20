# constexpr 的用法

constexpr 是 C++11 引入的一个关键字，用于声明编译时常量表达式。它允许在编译时计算表达式的值，从而提高程序的性能和安全性。
constexpr 可以用于变量、函数和构造函数，确保它们的值或行为在编译时确定。

## 1. constexpr 的基本用法

### 1.1 声明编译时常量
constexpr 可以用于声明编译时常量，这些常量的值在编译时确定，且不能被修改。
示例：
cpp复制
constexpr int MAX_SIZE = 100;  // 编译时常量
1.2 声明 constexpr 函数
constexpr 函数是可以在编译时计算结果的函数。它们必须满足以下条件：
函数体必须是单一的 return 语句（C++11）。
函数参数必须是常量表达式。
函数返回值必须是常量表达式。
从 C++14 开始，constexpr 函数的限制被放宽，允许包含多个语句。
示例：
cpp复制
constexpr int add(int a, int b) {
    return a + b;  // C++11
}

constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);  // C++14
}
1.3 声明 constexpr 构造函数
constexpr 构造函数允许在编译时构造对象。它们必须满足以下条件：
构造函数体必须是空的。
成员变量必须是 constexpr 或具有默认值。
从 C++14 开始，constexpr 构造函数的限制被放宽，允许包含多个语句。
示例：
cpp复制
struct Point {
    constexpr Point(int x, int y) : x_(x), y_(y) {}  // C++11
    int x_, y_;
};

constexpr Point origin(0, 0);  // 编译时构造对象
2. 示例代码
2.1 编译时常量
cpp复制
#include <iostream>

constexpr int MAX_SIZE = 100;

int main() {
    std::cout << "MAX_SIZE: " << MAX_SIZE << std::endl;
    return 0;
}
2.2 constexpr 函数
cpp复制
#include <iostream>

constexpr int add(int a, int b) {
    return a + b;
}

constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

int main() {
    constexpr int sum = add(5, 10);
    constexpr int fact = factorial(5);

    std::cout << "Sum: " << sum << std::endl;  // 输出：Sum: 15
    std::cout << "Factorial: " << fact << std::endl;  // 输出：Factorial: 120

    return 0;
}
2.3 constexpr 构造函数
cpp复制
#include <iostream>

struct Point {
    constexpr Point(int x, int y) : x_(x), y_(y) {}
    int x_, y_;
};

int main() {
    constexpr Point origin(0, 0);

    std::cout << "Origin: (" << origin.x_ << ", " << origin.y_ << ")" << std::endl;  // 输出：Origin: (0, 0)
    return 0;
}
3. constexpr 的注意事项
3.1 编译时常量表达式
constexpr 变量和函数的值必须在编译时确定。如果在运行时计算值，会导致编译错误。
3.2 constexpr 函数的限制
C++11：constexpr 函数必须是单一的 return 语句。
C++14：constexpr 函数可以包含多个语句。
C++20：constexpr 函数的限制进一步放宽，允许更多的复杂操作。
3.3 constexpr 构造函数的限制
C++11：constexpr 构造函数必须是空的。
C++14：constexpr 构造函数可以包含多个语句。
C++20：constexpr 构造函数的限制进一步放宽，允许更多的复杂操作。
4. 总结
constexpr 是一个非常强大的工具，用于声明编译时常量、函数和构造函数。它允许在编译时计算表达式的值，从而提高程序的性能和安全性。以下是 constexpr 的主要用法：
编译时常量：使用 constexpr 声明编译时常量。
constexpr 函数：声明可以在编译时计算结果的函数。
constexpr 构造函数：声明可以在编译时构造对象的构造函数。
通过合理使用 constexpr，可以提高代码的效率和安全性，同时减少运行时的计算开销。

关键字 constexpr 是C++11中引入的关键字，声明为constexpr类型的变量，编译器会验证该变量的值是否是一个常量表达式。
声明为constexpr的变量一定是一个常量，而且必须用常量表达式初始化：

```cpp
constexpr int mf = 0; // 0 是 常量表达式
constexpr int limit = mf + 2; // mf + 1 是常量表达式
constexpr int sz = size() // 只有当 size() 是一个 返回值为 constexpr 函数时才是一个正确的声明语句
```