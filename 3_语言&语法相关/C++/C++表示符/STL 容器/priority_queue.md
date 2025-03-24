# std::priority_queue

std::priority_queue 是一个优先队列，它基于堆（默认是最大堆）实现，用于存储元素并按优先级顺序访问它们。优先队列通常用于任务调度、事件处理等场景。

## 特点

### 元素顺序

+ 默认情况下，std::priority_queue 是一个最大堆，即队列顶部的元素是最大的。
+ 可以通过自定义比较函数来改变优先级顺序。

### 主要操作

+ push：将元素插入队列。
+ pop：移除队列顶部的元素。
+ top：访问队列顶部的元素。
+ empty：检查队列是否为空。
+ size：获取队列的大小。

### 示例代码

以下是一个使用 std::priority_queue 的示例：

```cpp
#include <iostream>
#include <queue>
#include <vector>
#include <functional> // for std::greater

int main() {
    // 默认最大堆
    std::priority_queue<int> maxHeap;

    // 插入元素
    maxHeap.push(10);
    maxHeap.push(30);
    maxHeap.push(20);

    // 访问和移除元素
    while (!maxHeap.empty()) {
        std::cout << maxHeap.top() << " ";
        maxHeap.pop();
    }
    std::cout << std::endl;

    // 最小堆
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;

    // 插入元素
    minHeap.push(10);
    minHeap.push(30);
    minHeap.push(20);

    // 访问和移除元素
    while (!minHeap.empty()) {
        std::cout << minHeap.top() << " ";
        minHeap.pop();
    }
    std::cout << std::endl;

    return 0;
}

// 输出
// 30 20 10
// 10 20 30
```

## 自定义比较函数

如果需要自定义优先级顺序，可以提供一个比较函数。例如，定义一个最小堆：

```cpp
struct Compare {
    bool operator()(int a, int b) {
        return a > b; // 小于号表示最小堆
    }
};

std::priority_queue<int, std::vector<int>, Compare> customHeap;
```

## 总结

std::priority_queue 是一个基于堆的优先队列，支持高效的插入和访问操作。
默认情况下，它是一个最大堆，但可以通过自定义比较函数来实现最小堆或其他优先级顺序。优先队列在任务调度、事件处理等场景中非常有用