# `std::optional` 的用法

`std::optional` 是 C++17 引入的一个模板类，用于表示一个可能有值也可能没有值的对象。它类似于 `nullptr`，但更安全、更灵活，避免了使用裸指针带来的潜在问题。`std::optional` 提供了一种优雅的方式来处理可选值，尤其是在函数返回值中。

## **1. `std::optional` 的基本用法**

### **1.1 包含头文件**

要使用 `std::optional`，需要包含头文件 `<optional>`：

```cpp
#include <optional>
```

### **1.2 声明 `std::optional`**

`std::optional` 是一个模板类，可以用于任何类型。声明时，需要指定模板参数。

**示例：**

```cpp
#include <iostream>
#include <optional>

int main() {
    std::optional<int> opt1;  // 默认构造，表示没有值
    std::optional<int> opt2(42);  // 构造时初始化为 42

    if (opt1) {
        std::cout << "opt1 has a value: " << *opt1 << std::endl;
    } else {
        std::cout << "opt1 has no value" << std::endl;
    }

    if (opt2) {
        std::cout << "opt2 has a value: " << *opt2 << std::endl;
    } else {
        std::cout << "opt2 has no value" << std::endl;
    }

    return 0;
}
```

### **1.3 检查是否有值**

使用 `has_value()` 或布尔上下文来检查 `std::optional` 是否有值。

**示例：**

```cpp
if (opt2.has_value()) {
    std::cout << "opt2 has a value: " << *opt2 << std::endl;
} else {
    std::cout << "opt2 has no value" << std::endl;
}
```

### **1.4 访问值**

使用 `*` 或 `value()` 方法访问 `std::optional` 中的值。如果 `std::optional` 没有值，调用 `value()` 会抛出异常。

**示例：**

```cpp
if (opt2) {
    std::cout << "opt2 has a value: " << opt2.value() << std::endl;
} else {
    std::cout << "opt2 has no value" << std::endl;
}
```

### **1.5 提供默认值**

使用 `value_or` 方法提供默认值，如果 `std::optional` 没有值，返回默认值。

**示例：**

```cpp
std::cout << "opt1 value or default: " << opt1.value_or(0) << std::endl;  // 输出：0
std::cout << "opt2 value or default: " << opt2.value_or(0) << std::endl;  // 输出：42
```

## **2. `std::optional` 的高级用法**

### **2.1 与函数返回值结合**

`std::optional` 常用于函数返回值，表示函数可能有返回值也可能没有返回值。

**示例：**

```cpp
#include <iostream>
#include <optional>

std::optional<int> findValue(const std::vector<int>& vec, int target) {
    for (int value : vec) {
        if (value == target) {
            return value;
        }
    }
    return {};  // 返回空的 optional
}

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto result = findValue(vec, 3);

    if (result) {
        std::cout << "Found value: " << *result << std::endl;
    } else {
        std::cout << "Value not found" << std::endl;
    }

    return 0;
}
```

### **2.2 与 `std::variant` 结合**

`std::optional` 可以与 `std::variant` 结合使用，表示一个变量可能有多种类型，也可能没有值。

**示例：**

```cpp
#include <iostream>
#include <optional>
#include <variant>

int main() {
    std::optional<std::variant<int, double, std::string>> opt;

    opt = 42;
    std::cout << "opt value: " << std::get<int>(*opt) << std::endl;

    opt = std::string("Hello, World!");
    std::cout << "opt value: " << std::get<std::string>(*opt) << std::endl;

    return 0;
}
```

## **3. 总结**

`std::optional` 是 C++17 引入的一个非常有用的工具，用于表示一个可能有值也可能没有值的对象。它提供了以下主要功能：

1. **声明 `std::optional`**：可以用于任何类型，表示一个可选值。
2. **检查是否有值**：使用 `has_value()` 或布尔上下文。
3. **访问值**：使用 `*` 或 `value()` 方法。
4. **提供默认值**：使用 `value_or` 方法。
5. **与函数返回值结合**：表示函数可能有返回值也可能没有返回值。
6. **与 `std::variant` 结合**：表示一个变量可能有多种类型，也可能没有值。

通过合理使用 `std::optional`，可以提高代码的安全性和可读性，避免使用裸指针带来的潜在问题。
