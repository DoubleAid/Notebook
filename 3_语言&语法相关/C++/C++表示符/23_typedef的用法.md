# typedef 的用法

typedef 是 C++ 中的一个关键字，用于为类型创建别名。它可以帮助简化复杂的类型声明，提高代码的可读性和可维护性。
虽然 C++11 引入了 using 语法来替代 typedef，但 typedef 仍然广泛使用，尤其是在一些旧代码中。

## 1. typedef 的基本用法

### 1.1 为类型创建别名

typedef 可以为任何类型创建一个别名，包括基本类型、类、结构体、枚举等。

语法：

`typedef existing_type new_type_name;`

示例：

```cpp
typedef int Integer;
typedef double Real;
typedef std::vector<int> IntVector;
typedef std::pair<int, std::string> IntStringPair;
```

### 1.2 为复杂类型创建别名

typedef 特别适用于简化复杂的类型声明，如指针、数组和函数指针。

示例：

```cpp
typedef int* IntPtr;  // 指针别名
typedef int Array[10];  // 数组别名
typedef void (*FuncPtr)(int);  // 函数指针别名
```

## 2. 示例代码

### 2.1 简化类型声明

```cpp
#include <iostream>
#include <vector>

typedef std::vector<int> IntVector;

int main() {
    IntVector vec = {1, 2, 3, 4, 5};

    for (int x : vec) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### 2.2 函数指针别名

```cpp
#include <iostream>

typedef void (*FuncPtr)(int);

void printNumber(int x) {
    std::cout << "Number: " << x << std::endl;
}

int main() {
    FuncPtr func = printNumber;
    func(42);

    return 0;
}
```

## 3. typedef 的高级用法

### 3.1 结构体和类的别名

typedef 可以为结构体或类创建别名，简化代码。

示例：

```cpp
struct Point {
    int x, y;
};

typedef Point PointType;

int main() {
    PointType p = {10, 20};
    std::cout << "Point: (" << p.x << ", " << p.y << ")" << std::endl;

    return 0;
}
```

### 3.2 枚举的别名

typedef 可以为枚举类型创建别名。

示例：

```cpp
enum Color { RED, GREEN, BLUE };

typedef Color ColorType;

int main() {
    ColorType c = GREEN;
    std::cout << "Color: " << c << std::endl;

    return 0;
}
```

## 4. typedef 与 using

C++11 引入了 using 语法，作为 typedef 的现代替代。using 语法更灵活，更符合现代 C++ 的风格。

示例：

```cpp
using Integer = int;
using Real = double;
using IntVector = std::vector<int>;
using FuncPtr = void (*)(int);
```

## 5. 总结

typedef 是一个非常有用的工具，用于为类型创建别名，简化复杂的类型声明。以下是 typedef 的主要用法：

+ 为类型创建别名：简化代码，提高可读性。
+ 简化复杂类型声明：如指针、数组和函数指针。
+ 结构体和类的别名：简化结构体或类的声明。
+ 枚举的别名：简化枚举类型的声明。

虽然 C++11 引入了 using 语法，但 typedef 仍然广泛使用，尤其是在一些旧代码中。通过合理使用 typedef 或 using，可以提高代码的可读性和可维护性。
