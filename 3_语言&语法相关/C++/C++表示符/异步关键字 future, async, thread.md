# 异步关键字

## std::future

### 1. std::future 是什么？

std::future 是 C++ 标准库中用于异步操作的结果管理的模板类。它允许线程在后台执行任务，而主线程可以等待任务完成并获取其结果。std::future 是异步编程的核心组件之一，通常与 std::async、std::thread 或线程池一起使用。

### 2. 功能和用途

std::future 的主要功能是提供一种机制，允许线程在后台执行任务，而主线程可以：

+ 等待任务完成：通过 std::future::get() 或 std::future::wait()。
+ 获取任务的返回值：通过 std::future::get()。
+ 检查任务是否完成：通过 std::future::valid() 和 std::future::wait_for()。
+ 取消任务：通过 std::future::wait_for() 或 std::future::wait_until()（如果任务支持取消）。

### 3. std::future 的主要成员函数

+ std::future::get()：阻塞当前线程，直到异步任务完成，并返回任务的结果。如果任务抛出异常，get() 会重新抛出异常。
+ std::future::wait()：阻塞当前线程，直到异步任务完成，但不获取结果。
+ std::future::wait_for()：阻塞当前线程，直到任务完成或超时。返回 std::future_status 枚举值，表示任务的状态（完成、超时或就绪）。
+ std::future::wait_until()：阻塞当前线程，直到任务完成或达到指定时间点。返回 std::future_status 枚举值。
+ std::future::valid()：检查 std::future 是否关联了一个有效的异步操作。

### 4. 应用场景

std::future 通常用于以下场景：
异步任务管理：
在后台线程中执行耗时任务，主线程继续执行其他工作，直到需要任务结果时再等待。
示例：

```cpp
std::future<int> result = std::async(std::launch::async, []() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 42;
});

std::cout << "Doing other work in main thread..." << std::endl;
int answer = result.get();  // 等待任务完成并获取结果
std::cout << "The answer is: " << answer << std::endl;
```

线程池任务管理：
在线程池中提交任务，并通过 std::future 获取任务结果。
你的线程池代码就是一个很好的例子：

```cpp
std::future<int> future = pool.enqueue(mul, 5, 6, i);
int result = future.get();  // 等待任务完成并获取结果
```

并发任务的同步：
等待多个异步任务完成，然后继续执行后续逻辑。
示例：

```cpp
std::vector<std::future<int>> futures;
for (int i = 0; i < 10; i++) {
    futures.push_back(std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return i * i;
    }));
}

for (auto& future : futures) {
    std::cout << "Result: " << future.get() << std::endl;
}
```

任务取消和超时处理：
使用 wait_for() 或 wait_until() 实现任务的超时处理。
示例：

```cpp
std::future<int> result = std::async(std::launch::async, []() {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return 42;
});

auto status = result.wait_for(std::chrono::seconds(2));
if (status == std::future_status::timeout) {
    std::cout << "Task timed out!" << std::endl;
} else {
    std::cout << "Result: " << result.get() << std::endl;
}
```

### 5. std::future 与 std::shared_future

+ std::future：每个 std::future 对象只能被等待一次（get() 或 wait()）。之后，它会失效，不能再次使用。
+ std::shared_future：允许多个线程共享同一个异步操作的结果。可以多次调用 get() 或 wait()，而不会失效。
示例：

```cpp
std::shared_future<int> shared_result = std::async(std::launch::async, []() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 42;
});

std::thread t1([shared_result]() {
    std::cout << "Thread 1: " << shared_result.get() << std::endl;
});

std::thread t2([shared_result]() {
    std::cout << "Thread 2: " << shared_result.get() << std::endl;
});

t1.join();
t2.join();
```

### 6. 总结

std::future 是一个强大的工具，用于管理异步任务的结果。它允许线程在后台执行任务，而主线程可以等待任务完成并获取结果。std::future 的主要功能包括：

+ 等待任务完成。
+ 获取任务的返回值。
+ 检查任务状态。
+ 实现超时处理。

它广泛应用于异步编程、线程池任务管理和并发任务的同步中。

## std::async

std::async 是 C++ 标准库中用于启动异步任务的函数。它提供了一种简单而强大的方式来执行异步操作，并通过 std::future 获取任务的结果。
std::async 是异步编程中的一个重要工具，尤其适合处理那些需要在后台线程中执行的任务。
std::async 的基本用法

### 1. 函数签名

std::async 的声明如下：

```cpp
template <typename Func, typename... Args>
std::future<typename std::result_of<Func(Args...)>::type>
async(std::launch policy, Func&& func, Args&&... args);
```

std::launch policy：启动策略，用于指定任务的执行方式。
Func&& func：要执行的函数或可调用对象。
Args&&... args：传递给函数的参数。

### 2. 启动策略

std::async 支持两种启动策略：

+ std::launch::async：强制任务在单独的线程中异步执行。
+ std::launch::deferred：任务会在调用 std::future::get() 或 std::future::wait() 时才执行，类似于延迟计算。

默认策略：如果未指定启动策略，std::async 会根据系统资源和任务情况选择 async 或 deferred。

### 3. 使用示例

#### (1) 异步执行任务

```cpp
#include <iostream>
#include <future>
#include <thread>
#include <chrono>

int compute(int value) {
    std::this_thread::sleep_for(std::chrono::seconds(2));  // 模拟耗时操作
    return value * value;
}

int main() {
    // 使用 std::async 启动异步任务
    std::future<int> result = std::async(std::launch::async, compute, 5);

    std::cout << "Doing other work in the main thread..." << std::endl;

    // 等待任务完成并获取结果
    int answer = result.get();
    std::cout << "The answer is: " << answer << std::endl;

    return 0;
}
```

#### (2) 延迟执行任务

```cpp
#include <iostream>
#include <future>
#include <chrono>

int compute(int value) {
    std::this_thread::sleep_for(std::chrono::seconds(2));  // 模拟耗时操作
    return value * value;
}

int main() {
    // 使用 std::async 启动延迟任务
    std::future<int> result = std::async(std::launch::deferred, compute, 5);

    std::cout << "Doing other work in the main thread..." << std::endl;

    // 调用 get() 时才会执行任务
    int answer = result.get();
    std::cout << "The answer is: " << answer << std::endl;

    return 0;
}
```

#### (3) 默认策略

```cpp
#include <iostream>
#include <future>
#include <chrono>

int compute(int value) {
    std::this_thread::sleep_for(std::chrono::seconds(2));  // 模拟耗时操作
    return value * value;
}

int main() {
    // 使用默认策略
    std::future<int> result = std::async(compute, 5);

    std::cout << "Doing other work in the main thread..." << std::endl;

    // 等待任务完成并获取结果
    int answer = result.get();
    std::cout << "The answer is: " << answer << std::endl;

    return 0;
}
```

### 4. 处理不同返回值类型

std::async 的返回值是一个 std::future 对象，其类型会根据任务的返回值类型自动推导。因此，你可以轻松处理不同类型的返回值。
例如：

```cpp
#include <iostream>
#include <future>
#include <chrono>
#include <string>

int computeInt(int value) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return value * value;
}

std::string computeString(int value) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return "Result: " + std::to_string(value);
}

int main() {
    // 异步执行返回 int 的任务
    std::future<int> futureInt = std::async(std::launch::async, computeInt, 5);
    // 异步执行返回 std::string 的任务
    std::future<std::string> futureString = std::async(std::launch::async, computeString, 10);

    // 等待并获取结果
    int resultInt = futureInt.get();
    std::string resultString = futureString.get();

    std::cout << "Int result: " << resultInt << std::endl;
    std::cout << "String result: " << resultString << std::endl;

    return 0;
}
```

### 5.异常处理

如果任务抛出异常，std::future::get() 会重新抛出该异常。因此，你需要在调用 get() 时处理异常：

```cpp
try {
    int result = futureInt.get();
    std::cout << "Result: " << result << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
}
```

### 6.总结

std::async：用于启动异步任务，返回一个 std::future 对象。
启动策略：

+ std::launch::async：强制任务在单独的线程中异步执行。
+ std::launch::deferred：任务在调用 get() 或 wait() 时才执行。
+ 默认策略：根据系统资源和任务情况选择执行方式。

优点：

+ 简单易用，适合简单的异步任务。
+ 自动管理线程资源。

缺点：

+ 不适合复杂的任务调度，例如任务队列或任务优先级。
+ 无法直接取消任务。

std::async 是 C++ 标准库中一个非常强大的工具，尤其适合处理简单的异步任务。如果你需要更复杂的任务调度和管理，可以考虑使用线程池或其他并发库。

## std::thread

std::thread 是 C++ 标准库中用于表示线程的类，它是 C++11 引入的线程支持库的核心组件之一。
std::thread 提供了一种简单而强大的方式来创建和管理线程，使得多线程编程更加直观和安全。

### std::thread 的主要功能

+ 创建线程：通过构造函数启动一个新线程。
+ 线程同步：提供 join() 和 detach() 方法来同步线程的执行。
+ 线程管理：提供 id 和 hardware_concurrency() 等工具来管理线程。

### std::thread 的构造函数

```cpp
template <typename Func, typename... Args>
explicit thread(Func&& f, Args&&... args);
```

Func&& f：要在线程中执行的函数或可调用对象。
Args&&... args：传递给函数的参数。

### 主要成员函数

+ join()：阻塞当前线程，直到被调用的线程完成执行。如果线程已经完成执行，则不会阻塞。如果线程被分离（detach()），则不能调用 join()。

示例：

```cpp
std::thread t([]() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Thread finished!" << std::endl;
});
t.join();  // 等待线程完成
```

+ detach()：将线程分离为后台线程，即使创建线程的线程已经结束执行。分离的线程会独立运行，即使创建它的线程已经退出。

示例：

```cpp
std::thread t([]() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Detached thread finished!" << std::endl;
});
t.detach();  // 分离线程
```

+ joinable()：检查线程是否可以被 join() 或 detach()。返回 true 如果线程正在运行且尚未被分离或加入。

示例：

```cpp
if (t.joinable()) {
    t.join();
}
```

+ get_id()：获取线程的唯一标识符。

示例：

```cpp
std::thread::id id = std::this_thread::get_id();
std::cout << "Thread ID: " << id << std::endl;
```

+ std::this_thread::sleep_for()：让当前线程暂停指定的时间。

示例：

```cpp
std::this_thread::sleep_for(std::chrono::seconds(1));
```

### 使用示例

#### (1) 创建和同步线程

```cpp
#include <iostream>
#include <thread>
#include <chrono>

void workerFunction(int id) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Thread " << id << " finished!" << std::endl;
}

int main() {
    std::thread t1(workerFunction, 1);
    std::thread t2(workerFunction, 2);

    t1.join();  // 等待线程 t1 完成
    t2.join();  // 等待线程 t2 完成

    std::cout << "All threads finished!" << std::endl;
    return 0;
}
```

#### (2) 分离线程

```cpp
#include <iostream>
#include <thread>
#include <chrono>

void workerFunction() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Detached thread finished!" << std::endl;
}

int main() {
    std::thread t(workerFunction);
    t.detach();  // 分离线程

    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Main thread finished!" << std::endl;

    return 0;  // 主线程退出，但分离的线程仍在运行
}
```

#### (3) 使用 Lambda 表达式

```cpp
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::thread t([]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "Lambda thread finished!" << std::endl;
    });

    t.join();
    std::cout << "Main thread finished!" << std::endl;
    return 0;
}
```

### 注意事项

+ 线程同步：如果线程未被 join() 或 detach()，程序结束时会调用线程的析构函数，这会导致程序崩溃（抛出 std::terminate 异常）。确保在程序退出前正确同步线程。
+ 线程安全：多线程程序需要考虑线程安全问题，例如数据竞争和死锁。使用互斥锁（std::mutex）或其他同步机制来保护共享数据。
+ 线程数量：使用 std::thread::hardware_concurrency() 获取系统支持的最大线程数，以优化线程的使用。

### 总结

std::thread 是 C++ 标准库中用于创建和管理线程的类，提供了简单而强大的接口来支持多线程编程。它的主要功能包括：

+ 创建线程。
+ 同步线程（join() 和 detach()）。
+ 管理线程（获取线程 ID、检查线程状态等）。

通过合理使用 std::thread，可以实现高效的并发编程。

## std::promise
