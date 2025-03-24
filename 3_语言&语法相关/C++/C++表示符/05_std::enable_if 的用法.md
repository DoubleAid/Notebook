# `std::enable_if` 的用法

`std::enable_if` 是 C++11 引入的一个模板元编程工具，用于在编译时根据条件启用或禁用模板函数或类。它通常用于实现 SFINAE（Substitution Failure Is Not An Error）特性，即在模板实例化时，根据条件选择合适的函数或类模板。

## **1. `std::enable_if` 的基本用法**

`std::enable_if` 的语法如下：

```cpp
template <bool B, typename T = void>
struct enable_if;

template <typename T>
struct enable_if<true, T> { using type = T; };
```

- **`B`**：布尔条件，决定是否启用模板。
- **`T`**：默认类型，通常为 `void`，但可以指定其他类型。

## **2. 示例代码**

### **2.1 启用特定类型的函数**

假设你有一个函数模板，希望它只对特定类型生效。可以使用 `std::enable_if` 来实现。

```cpp
#include <iostream>
#include <type_traits>

template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
void print(T value) {
    std::cout << "Integral value: " << value << std::endl;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
void print(T value) {
    std::cout << "Floating point value: " << value << std::endl;
}

int main() {
    print(10);    // 输出：Integral value: 10
    print(3.14);  // 输出：Floating point value: 3.14
    return 0;
}
```

### **2.2 启用特定条件的函数**

假设你希望函数只在某个条件满足时生效。可以使用 `std::enable_if` 来实现。

```cpp
#include <iostream>
#include <type_traits>

template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
void process(T value) {
    std::cout << "Processing integral value: " << value << std::endl;
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
void process(T value) {
    std::cout << "Processing non-integral value: " << value << std::endl;
}

int main() {
    process(10);    // 输出：Processing integral value: 10
    process(3.14);  // 输出：Processing non-integral value: 3.14
    return 0;
}
```

## **3. 使用 `std::enable_if_t`（C++17）**

从 C++17 开始，`std::enable_if` 的使用变得更加简洁，引入了 `std::enable_if_t`。

```cpp
#include <iostream>
#include <type_traits>

template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
void print(T value) {
    std::cout << "Integral value: " << value << std::endl;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
void print(T value) {
    std::cout << "Floating point value: " << value << std::endl;
}

int main() {
    print(10);    // 输出：Integral value: 10
    print(3.14);  // 输出：Floating point value: 3.14
    return 0;
}
```

## **4. 总结**

`std::enable_if` 是一个强大的工具，用于在编译时根据条件启用或禁用模板函数或类。它通过 SFINAE 特性实现，允许你根据类型或条件选择合适的模板实例。以下是 `std::enable_if` 的主要用法：

1. **启用特定类型的函数**：通过 `std::enable_if` 检查类型是否满足特定条件。
2. **启用特定条件的函数**：通过 `std::enable_if` 检查条件是否为真。
3. **使用 `std::enable_if_t`**：从 C++17 开始，`std::enable_if_t` 提供了更简洁的语法。

通过合理使用 `std::enable_if`，可以实现更灵活、更强大的模板编程。
