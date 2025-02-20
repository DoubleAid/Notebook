# generate 和 generate_n 的用法

std::generate 和 std::generate_n 是 C++ 标准库中的算法，用于生成序列中的值。它们的主要区别在于 std::generate 用于整个范围，而 std::generate_n 用于指定数量的元素。以下是它们的详细用法和示例。

## 1. std::generate 的用法

std::generate 用于将一个生成器函数生成的值分配给指定范围内的所有元素。
语法

```cpp
template <class ForwardIterator, class Generator>
void generate(ForwardIterator first, ForwardIterator last, Generator gen);
```

+ first：指向范围开始的前向迭代器。
+ last：指向范围结束的前向迭代器。
+ gen：一个无参数的函数对象，用于生成值。

示例

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec(10);

    // 使用 std::generate 填充向量
    std::generate(vec.begin(), vec.end(), []() {
        static int counter = 0;
        return counter++;
    });

    // 输出结果
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

// 输出：
// 0 1 2 3 4 5 6 7 8 9
```

## 2. std::generate_n 的用法

std::generate_n 用于将一个生成器函数生成的值分配给指定数量的元素。
语法

```cpp
template <class OutputIterator, class Size, class Generator>
OutputIterator generate_n(OutputIterator first, Size n, Generator gen);
```

+ first：指向范围开始的输出迭代器。
+ n：要生成的元素数量。
+ gen：一个无参数的函数对象，用于生成值。

返回值：指向最后一个生成元素之后的迭代器。
示例

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec(10);

    // 使用 std::generate_n 填充向量的前 5 个元素
    std::generate_n(vec.begin(), 5, []() {
        static int counter = 0;
        return counter++;
    });

    // 输出结果
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

// 输出：
// 0 1 2 3 4 0 0 0 0 0
```

## 3. 使用 Lambda 表达式

std::generate 和 std::generate_n 常与 Lambda 表达式结合使用，以实现更灵活的值生成。
示例

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec(10);

    // 使用 Lambda 表达式生成随机数
    std::generate(vec.begin(), vec.end(), []() {
        return std::rand() % 100;  // 生成 0 到 99 的随机数
    });

    // 输出结果
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

// 输出示例：
// 42 67 23 89 12 56 78 34 90 11
```

## 4. 总结

std::generate：用于整个范围，生成器函数为范围内的每个元素生成值。
std::generate_n：用于指定数量的元素，生成器函数为指定数量的元素生成值。
Lambda 表达式：常与 std::generate 和 std::generate_n 结合使用，实现灵活的值生成。
通过合理使用这些算法，可以简化代码，提高效率。
