# list

在 C++ 中，`std::list` 是一个双向链表容器，提供了高效的插入和删除操作。`std::list` 是标准模板库（STL）的一部分，提供了丰富的成员函数来操作链表。

## `std::list` 的基本用法

### 定义和初始化

```cpp
#include <list>
#include <iostream>

int main() {
    std::list<int> myList;

    // 添加元素
    myList.push_back(10);  // 在链表末尾添加元素
    myList.push_front(20); // 在链表头部添加元素

    // 初始化列表
    std::list<int> myList2 = {1, 2, 3, 4, 5};

    // 使用迭代器
    for (auto it = myList2.begin(); it != myList2.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

## `std::list` 的成员函数

### 1. **构造函数**

- `std::list()`: 默认构造函数，创建一个空链表。
- `std::list(size_type count, const T& value)`: 创建一个包含 `count` 个 `value` 的链表。
- `std::list(const std::list& other)`: 拷贝构造函数。
- `std::list(std::list&& other)`: 移动构造函数。
- `std::list(const Allocator& alloc)`: 使用指定分配器构造空链表。
- `std::list(size_type count, const T& value, const Allocator& alloc)`: 使用指定分配器构造链表。
- `std::list(const std::list& other, const Allocator& alloc)`: 使用指定分配器拷贝构造链表。
- `std::list(std::list&& other, const Allocator& alloc)`: 使用指定分配器移动构造链表。
- `std::list(std::initializer_list<value_type> init)`: 使用初始化列表构造链表。
- `std::list(std::initializer_list<value_type> init, const Allocator& alloc)`: 使用初始化列表和指定分配器构造链表。

### 2. **赋值操作**

- `operator=`: 赋值运算符，将一个链表的内容赋给另一个链表。
- `assign()`: 将链表的内容替换为新的内容。
- `get_allocator()`: 返回链表使用的分配器。

### 3. **迭代器**

- `begin()`: 返回指向链表第一个元素的迭代器。
- `end()`: 返回指向链表最后一个元素之后的迭代器。
- `rbegin()`: 返回指向链表最后一个元素的反向迭代器。
- `rend()`: 返回指向链表第一个元素之前的反向迭代器。
- `cbegin()`: 返回指向链表第一个元素的常量迭代器。
- `cend()`: 返回指向链表最后一个元素之后的常量迭代器。
- `crbegin()`: 返回指向链表最后一个元素的常量反向迭代器。
- `crend()`: 返回指向链表第一个元素之前的常量反向迭代器。

### 4. **容量**

- `empty()`: 检查链表是否为空。
- `size()`: 返回链表中的元素数量。
- `max_size()`: 返回链表可以容纳的最大元素数量。

### 5. **修改器**

- `clear()`: 清空链表。
- `insert()`: 在指定位置插入元素。
- `erase()`: 删除指定位置的元素。
- `push_front()`: 在链表头部插入元素。
- `pop_front()`: 删除链表头部的元素。
- `push_back()`: 在链表末尾插入元素。
- `pop_back()`: 删除链表末尾的元素。
- `resize()`: 改变链表的大小。
- `swap()`: 交换两个链表的内容。

### 6. **操作**

- `splice()`: 将另一个链表的元素移动到当前链表。
- `remove()`: 删除链表中满足特定条件的元素。
- `remove_if()`: 删除链表中满足特定条件的元素。
- `unique()`: 删除链表中连续的重复元素。
- `sort()`: 对链表中的元素进行排序。
- `merge()`: 将两个有序链表合并为一个有序链表。
- `reverse()`: 反转链表。
- `splice_after()`: 将另一个链表的元素移动到当前链表的指定位置之后。
- `merge_after()`: 将两个有序链表合并为一个有序链表，插入到指定位置之后。
- `sort_after()`: 对链表中的元素进行排序，从指定位置之后开始。

## `splice` 的作用

`splice` 是 `std::list` 的一个成员函数，用于将另一个链表的元素移动到当前链表中。它不会复制元素，而是直接移动元素，因此效率很高。

### 语法

```cpp
void splice(iterator position, list& x);
void splice(iterator position, list&& x);
void splice(iterator position, list& x, iterator i);
void splice(iterator position, list&& x, iterator i);
void splice(iterator position, list& x, iterator first, iterator last);
void splice(iterator position, list&& x, iterator first, iterator last);
```

### 示例

```cpp
#include <list>
#include <iostream>

int main() {
    std::list<int> list1 = {1, 2, 3};
    std::list<int> list2 = {4, 5, 6};

    // 将 list2 的所有元素移动到 list1 的末尾
    list1.splice(list1.end(), list2);

    // 输出结果
    for (int value : list1) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}

// 输出
// 1 2 3 4 5 6
```

## 总结

`std::list` 是一个双向链表容器，提供了高效的插入和删除操作。`splice` 函数用于将另一个链表的元素移动到当前链表中，效率很高。`std::list` 还提供了丰富的成员函数，用于操作链表，如 `insert`、`erase`、`push_front`、`pop_front`、`push_back`、`pop_back`、`resize`、`swap`、`remove`、`remove_if`、`unique`、`sort`、`merge`、`reverse` 等。