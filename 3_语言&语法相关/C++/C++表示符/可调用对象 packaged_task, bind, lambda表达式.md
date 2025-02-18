# 可调用对象

## std::packaged_task

`std::packaged_task` 是 C++ 标准库中用于将可调用对象（如函数、Lambda 表达式、函数对象等）包装起来，并与一个 std::future 对象关联的工具。
它允许你在后台线程中执行任务，并通过 std::future 获取任务的结果。std::packaged_task 是实现异步编程的重要组件之一，尤其是在线程池和任务队列的实现中。

### std::packaged_task 的功能和用途

+ 包装可调用对象：将函数、Lambda 表达式或其他可调用对象包装起来，以便在其他线程中执行。
+ 与 std::future 关联：允许调用者通过 std::future 获取任务的结果。
+ 支持异步任务：常用于线程池、任务队列或其他异步编程场景。

### std::packaged_task 的构造函数

std::packaged_task 的构造函数如下：

```cpp
template <typename Func>
packaged_task(Func&& func);
```

Func 是一个可调用对象，可以是函数指针、Lambda 表达式、函数对象等。
包装后的任务可以被调用多次，但每次调用都会返回相同的结果。

### std::packaged_task 的主要成员函数

+ operator()：调用包装的可调用对象。可以在任何线程中调用，结果会通过关联的 std::future 返回。
+ get_future()：返回与任务关联的 std::future 对象，用于获取任务的结果。每个 std::packaged_task 只能调用一次 get_future()。
+ valid()：检查 std::packaged_task 是否关联了有效的任务。如果任务尚未被调用或尚未完成，返回 true。

### 使用示例

#### (1) 基本用法

以下是一个简单的例子，展示如何使用 std::packaged_task 包装一个函数，并通过 std::future 获取结果：

```cpp
#include <iostream>
#include <thread>
#include <future>
#include <packaged_task>
#include <functional>

int compute(int value) {
    std::this_thread::sleep_for(std::chrono::seconds(2));  // 模拟耗时操作
    return value * value;
}

int main() {
    // 创建一个 std::packaged_task 对象，包装 compute 函数
    std::packaged_task<int(int)> task(compute);

    // 获取与任务关联的 std::future 对象
    std::future<int> result = task.get_future();

    // 在后台线程中调用任务
    std::thread worker(std::move(task), 5);

    std::cout << "Doing other work in the main thread..." << std::endl;

    // 等待任务完成并获取结果
    int answer = result.get();
    std::cout << "The answer is: " << answer << std::endl;

    worker.join();
    return 0;
}
```

#### (2) 使用 Lambda 表达式

std::packaged_task 也可以包装 Lambda 表达式：

```cpp
#include <iostream>
#include <thread>
#include <future>
#include <packaged_task>
#include <functional>

int main() {
    // 创建一个 std::packaged_task 对象，包装 Lambda 表达式
    std::packaged_task<int()> task([]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));  // 模拟耗时操作
        return 42;
    });

    // 获取与任务关联的 std::future 对象
    std::future<int> result = task.get_future();

    // 在后台线程中调用任务
    std::thread worker(std::move(task));

    std::cout << "Doing other work in the main thread..." << std::endl;

    // 等待任务完成并获取结果
    int answer = result.get();
    std::cout << "The answer is: " << answer << std::endl;

    worker.join();
    return 0;
}
```

#### (3) 在线程池中使用

std::packaged_task 常用于线程池的实现中。以下是一个简单的线程池示例，展示如何使用 std::packaged_task：

```cpp
#include <iostream>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>

class ThreadPool {
public:
    ThreadPool(size_t thread_count);
    ~ThreadPool();

    template <typename Func, typename... Args>
    auto enqueue(Func&& func, Args&&... args) -> std::future<typename std::result_of<Func(Args...)>::type>;

private:
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> threads_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
};

ThreadPool::ThreadPool(size_t thread_count) {
    for (size_t i = 0; i < thread_count; ++i) {
        threads_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

template <typename Func, typename... Args>
auto ThreadPool::enqueue(Func&& func, Args&&... args) -> std::future<typename std::result_of<Func(Args...)>::type> {
    using ReturnType = typename std::result_of<Func(Args...)>::type;
    std::packaged_task<ReturnType()> task(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));
    std::future<ReturnType> result = task.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        tasks_.emplace([task = std::move(task)]() mutable { task(); });
    }
    condition_.notify_one();
    return result;
}

int main() {
    ThreadPool pool(4);

    auto future1 = pool.enqueue([](int a, int b) { return a + b; }, 5, 10);
    auto future2 = pool.enqueue([](int a, int b) { return a * b; }, 5, 10);

    std::cout << "Result 1: " << future1.get() << std::endl;
    std::cout << "Result 2: " << future2.get() << std::endl;

    return 0;
}
```

### 总结

std::packaged_task：用于将可调用对象包装起来，并与 std::future 关联，以便在后台线程中执行任务。

#### 主要用途

+ 实现异步任务。
+ 用于线程池和任务队列。

#### 优点

+ 提供了灵活的任务包装和结果获取机制。
+ 与 std::future 和 std::promise 配合使用，可以实现复杂的异步编程模式。

#### 缺点

+ 使用相对复杂，需要理解 std::future 和 std::promise 的机制。
+ std::packaged_task 是 C++ 标准库中一个非常强大的工具，尤其适合需要在后台线程中执行任务并获取结果的场景。
