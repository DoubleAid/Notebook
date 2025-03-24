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

## bind

std::bind 是 C++ 标准库中的一个函数绑定器，用于创建可调用对象，这些对象可以将参数绑定到函数、lambda 表达式或函数对象上。
std::bind 的主要用途是固定某些参数，从而简化函数调用或适配函数接口。

### 1. std::bind 的基本用法

std::bind 的语法如下：

```cpp
template <typename F, typename... Args>
std::bind(F&& f, Args&&... args);
```

+ F&& f：要绑定的可调用对象（如函数指针、lambda 表达式、函数对象等）。
+ Args&&... args：要绑定的参数。

std::bind 返回一个 std::function 对象，该对象可以被调用，并将绑定的参数传递给目标函数。

### 2. 示例用法

#### 示例 1：绑定函数参数

假设有一个函数：

```cpp
void print_message(const std::string& msg) {
    std::cout << msg << std::endl;
}
```

可以使用 std::bind 将参数绑定到该函数：

```cpp
#include <iostream>
#include <functional>
#include <string>

void print_message(const std::string& msg) {
    std::cout << msg << std::endl;
}

int main() {
    auto print_hello = std::bind(print_message, "Hello, World!");
    print_hello();  // 输出：Hello, World!
    return 0;
}
```

#### 示例 2：绑定成员函数

假设有一个类，包含一个成员函数：

```cpp
class Printer {
public:
    void print(const std::string& msg) const {
        std::cout << msg << std::endl;
    }
};
```

可以使用 std::bind 将成员函数绑定到对象上：

```cpp
#include <iostream>
#include <functional>
#include <string>

class Printer {
public:
    void print(const std::string& msg) const {
        std::cout << msg << std::endl;
    }
};

int main() {
    Printer printer;
    auto print_hello = std::bind(&Printer::print, &printer, "Hello, World!");
    print_hello();  // 输出：Hello, World!
    return 0;
}
```

#### 示例 3：绑定参数到函数对象

假设有一个函数对象：

```cpp
struct Adder {
    int operator()(int a, int b) const {
        return a + b;
    }
};
```

可以使用 std::bind 将参数绑定到函数对象上：

```cpp
#include <iostream>
#include <functional>

struct Adder {
    int operator()(int a, int b) const {
        return a + b;
    }
};

int main() {
    Adder adder;
    auto add_5 = std::bind(adder, 5, std::placeholders::_1);
    std::cout << add_5(10) << std::endl;  // 输出：15
    return 0;
}
```

### 3. 使用占位符

std::bind 支持占位符（std::placeholders::_1, std::placeholders::_2 等），用于在绑定时保留某些参数的位置。这些占位符可以在调用时动态传递参数。

示例：使用占位符

```cpp
#include <iostream>
#include <functional>

void print_message(const std::string& prefix, const std::string& msg) {
    std::cout << prefix << ": " << msg << std::endl;
}

int main() {
    auto print_hello = std::bind(print_message, "Hello", std::placeholders::_1);
    print_hello("World!");  // 输出：Hello: World!
    return 0;
}
```

### 4. std::bind 的优势和限制

**优势**

+ 固定参数：可以将某些参数固定，简化函数调用。
+ 适配接口：可以将函数适配到需要特定参数的接口。
+ 灵活性：支持绑定函数、成员函数、函数对象和 lambda 表达式。

**限制**

+ 类型推导复杂：std::bind 的类型推导较为复杂，可能导致编译错误。
+ 性能开销：std::bind 返回的是 std::function 对象，可能涉及额外的动态分配。
+ 可读性差：在复杂场景中，std::bind 的代码可能难以理解。

### 5. 替代方案

在现代 C++ 中，std::bind 的功能可以通过 lambda 表达式更简洁地实现。例如：

```cpp
auto print_hello = [](const std::string& msg) {
    print_message("Hello", msg);
};
print_hello("World!");  // 输出：Hello: World!
```

### 6. 总结

std::bind 是一个强大的工具，用于绑定函数参数、适配接口和简化函数调用。它支持占位符，允许在调用时动态传递参数。
然而，在现代 C++ 中，lambda 表达式通常是一个更简洁、更高效的替代方案。

## lambda 表达式

Lambda 表达式是 C++11 引入的一种匿名函数对象，它提供了一种简洁的方式来定义内联函数。Lambda 表达式在 C++ 中非常强大且灵活，
广泛应用于标准库（如算法库）和现代 C++ 编程中。以下是 Lambda 表达式的种类和常见用法。

### 1. Lambda 表达式的基本语法

Lambda 表达式的语法如下：

```cpp
[capture-list] (parameters) -> return-type { body }
[capture-list]：捕获列表，用于捕获当前作用域中的变量。
(parameters)：参数列表，指定 lambda 表达式的输入参数。
-> return-type：返回类型（可选），显式指定返回类型。
{ body }：函数体，包含 lambda 表达式的实现。
```

### 2. Lambda 表达式的种类和用法

#### 2.1 无捕获的 Lambda 表达式

无捕获的 lambda 表达式不捕获当前作用域中的任何变量。
示例：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    int sum = 0;

    std::for_each(vec.begin(), vec.end(), [&sum](int x) {
        sum += x;
    });

    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
```

#### 2.2 捕获变量的 Lambda 表达式

Lambda 表达式可以通过捕获列表捕获当前作用域中的变量。捕获方式有两种：
按值捕获：捕获变量的拷贝。
按引用捕获：捕获变量的引用。
示例：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    int sum = 0;

    // 按引用捕获 sum
    std::for_each(vec.begin(), vec.end(), [&sum](int x) {
        sum += x;
    });

    std::cout << "Sum (by reference): " << sum << std::endl;

    // 按值捕获 sum
    int local_sum = 0;
    std::for_each(vec.begin(), vec.end(), [local_sum](int x) mutable {
        local_sum += x;
    });

    std::cout << "Sum (by value): " << local_sum << std::endl;

    return 0;
}
```

#### 2.3 带返回值的 Lambda 表达式

Lambda 表达式可以有返回值，返回类型可以显式指定，也可以通过 auto 推导。
示例：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 带返回值的 lambda 表达式
    auto is_even = [](int x) -> bool {
        return x % 2 == 0;
    };

    int count = std::count_if(vec.begin(), vec.end(), is_even);
    std::cout << "Number of even elements: " << count << std::endl;

    return 0;
}
```

#### 2.4 带参数的 Lambda 表达式

Lambda 表达式可以接受参数，参数列表类似于普通函数的参数列表。
示例：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 带参数的 lambda 表达式
    auto add = [](int a, int b) {
        return a + b;
    };

    int result = std::accumulate(vec.begin(), vec.end(), 0, add);
    std::cout << "Sum: " << result << std::endl;

    return 0;
}
```

#### 2.5 mutable Lambda 表达式

默认情况下，lambda 表达式的捕获变量是不可变的。如果需要修改捕获的变量，可以使用 mutable 关键字。
示例：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    int local_sum = 0;

    // mutable 允许修改捕获的变量
    std::for_each(vec.begin(), vec.end(), [local_sum](int x) mutable {
        local_sum += x;
    });

    std::cout << "Sum (mutable): " << local_sum << std::endl;

    return 0;
}
```

#### 2.6 无参数的 Lambda 表达式

Lambda 表达式可以没有参数，这种情况下可以用于延迟计算或封装操作。
示例：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    int x = 10;
    auto print_x = [x]() {
        std::cout << "x: " << x << std::endl;
    };

    print_x();  // 输出：x: 10

    return 0;
}
```

#### 2.7 Lambda 表达式作为函数对象

Lambda 表达式可以作为函数对象，存储在 std::function 或其他函数对象容器中。
示例：

```cpp
#include <iostream>
#include <functional>

int main() {
    auto add = [](int a, int b) {
        return a + b;
    };

    std::function<int(int, int)> func = add;
    std::cout << "Result: " << func(5, 10) << std::endl;

    return 0;
}
```

#### 2.8 Lambda 表达式与标准库

Lambda 表达式广泛应用于标准库的算法（如 std::for_each、std::sort、std::transform 等）。
示例：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 使用 lambda 表达式排序
    std::sort(vec.begin(), vec.end(), [](int a, int b) {
        return a > b;
    });

    for (int x : vec) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### 3. 总结

Lambda 表达式是 C++ 中一种非常强大的工具，具有以下特点：
简洁性：可以快速定义匿名函数对象。
灵活性：支持捕获变量、参数列表、返回值和 mutable。
适用性：广泛应用于标准库和现代 C++ 编程。
通过合理使用 Lambda 表达式，可以简化代码，提高可读性和效率。

## result_of

## invoke_result

std::invoke_result 是 C++17 引入的一个模板类，用于推导调用可调用对象（如函数、lambda 表达式、函数对象等）时的返回类型。
它是对 std::result_of 的改进，并在 C++20 中完全替代了 std::result_of。

### 1. std::invoke_result 的基本用法

std::invoke_result 的语法如下：

```cpp
template <typename Callable, typename... Args>
class std::invoke_result;
```

+ Callable：可调用对象的类型（如函数指针、lambda 表达式、函数对象等）。
+ Args...：调用时传递的参数类型。

std::invoke_result 提供了一个嵌套类型 type，表示调用 Callable 时的返回类型。
获取返回类型
通过 std::invoke_result_t（std::invoke_result 的别名）可以更方便地获取返回类型：

```cpp
using ResultType = std::invoke_result_t<Callable, Args...>;
```

### 2. 示例用法

#### 示例 1：推导函数的返回类型

假设有一个函数：

```cpp
int add(int a, int b) {
    return a + b;
}
```

可以使用 std::invoke_result 推导其返回类型：

```cpp
using ResultType = std::invoke_result_t<decltype(add), int, int>;
static_assert(std::is_same_v<ResultType, int>, "The result type should be int");
```

#### 示例 2：推导 lambda 表达式的返回类型

```cpp
auto lambda = [](int x, int y) -> double { return x + y; };
using LambdaResultType = std::invoke_result_t<decltype(lambda), int, int>;
static_assert(std::is_same_v<LambdaResultType, double>, "The result type should be double");
```

#### 示例 3：推导函数对象的返回类型

```cpp
struct Adder {
    int operator()(int a, int b) const { return a + b; }
};
Adder adder;
using AdderResultType = std::invoke_result_t<decltype(adder), int, int>;
static_assert(std::is_same_v<AdderResultType, int>, "The result type should be int");
```

### 3. 使用 std::invoke_result 的场景

#### 场景 1：线程池中的任务提交

在你的线程池代码中，可以使用 std::invoke_result 来推导任务的返回类型：

```cpp
template <typename Func, typename... Args>
auto ThreadPool::enqueue(Func&& func, Args&&... args) -> std::future<std::invoke_result_t<Func, Args...>> {
    using ResultType = std::invoke_result_t<Func, Args...>;
    std::packaged_task<ResultType()> task(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));
    std::future<ResultType> result = task.get_future();
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        jobs_.emplace([task = std::move(task)]() mutable { task(); });
    }
    jobs_available_.notify_one();
    return result;
}
```

#### 场景 2：检查函数返回类型

在模板编程中，std::invoke_result 可以用于检查函数的返回类型是否符合预期：

```cpp
template <typename F, typename... Args>
constexpr bool check_return_type() {
    return std::is_same_v<std::invoke_result_t<F, Args...>, int>;
}

static_assert(check_return_type<decltype(add), int, int>(), "Function should return int");
```

### 4. 注意事项

std::invoke_result 的限制：
std::invoke_result 只能推导可调用对象的返回类型，不能检查调用是否有效（例如，是否会发生异常）。
如果调用是无效的（如参数类型不匹配），编译器会报错。
替代方案：
如果需要更复杂的调用检查，可以使用 std::is_invocable 和 std::is_invocable_r，这些类型特性可以检查调用是否有效以及返回类型是否符合预期。

### 5. 总结

std::invoke_result 是一个强大的工具，用于推导可调用对象的返回类型。它在 C++17 中引入，并在 C++20 中替代了 std::result_of。通过 std::invoke_result_t，可以更方便地获取返回类型，适用于线程池、模板编程等多种场景。
希望这些示例和解释能帮助你更好地理解和使用 std::invoke_result！
