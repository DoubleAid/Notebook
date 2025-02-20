# atomic 的用法

std::atomic 是 C++11 引入的一个模板类，用于提供原子操作，确保在多线程环境中对变量的访问和修改是线程安全的。
std::atomic 支持多种原子操作，包括读取、写入、比较交换（CAS）和算术操作。

## 1. std::atomic 的基本用法

std::atomic 的语法如下：

```cpp
#include <atomic>
std::atomic<T> atomic_var;
```

T：可以是任何基本类型（如 int、float、bool）或指针类型。
atomic_var：一个原子变量，对它的操作都是原子的。

## 2. 常见操作

### 2.1 原子读取和写入

std::atomic 提供了 load 和 store 方法，用于原子读取和写入变量的值。
示例：

```cpp
#include <atomic>
#include <iostream>
#include <thread>

std::atomic<int> counter(0);

void increment() {
    for (int i = 0; i < 100000; ++i) {
        counter.fetch_add(1, std::memory_order_relaxed);  // 原子加 1
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Counter: " << counter.load() << std::endl;  // 原子读取
    return 0;
}
```

### 2.2 比较交换（CAS）

std::atomic 提供了 compare_exchange_strong 和 compare_exchange_weak 方法，用于原子比较和交换操作。
示例：

```cpp
#include <atomic>
#include <iostream>

std::atomic<int> value(0);

void cas_example() {
    int expected = 0;
    int desired = 1;

    if (value.compare_exchange_strong(expected, desired)) {
        std::cout << "CAS succeeded, value is now " << desired << std::endl;
    } else {
        std::cout << "CAS failed, value is " << value.load() << std::endl;
    }
}

int main() {
    cas_example();
    return 0;
}
```

### 2.3 算术操作
std::atomic 提供了多种原子算术操作，如 fetch_add、fetch_sub、fetch_and、fetch_or 等。
示例：

```cpp
#include <atomic>
#include <iostream>

std::atomic<int> counter(0);

void increment() {
    counter.fetch_add(1, std::memory_order_relaxed);  // 原子加 1
}

void decrement() {
    counter.fetch_sub(1, std::memory_order_relaxed);  // 原子减 1
}

int main() {
    increment();
    decrement();

    std::cout << "Counter: " << counter.load() << std::endl;  // 原子读取
    return 0;
}
```

## 3. 内存顺序（Memory Order）

std::atomic 的操作可以指定内存顺序，用于控制操作的可见性和顺序。常见的内存顺序包括：

+ std::memory_order_relaxed：最弱的内存顺序，不保证操作的可见性。
+ std::memory_order_acquire：用于加载操作，确保后续操作可见。
+ std::memory_order_release：用于存储操作，确保先前操作可见。
+ std::memory_order_acq_rel：用于读写操作，结合了 acquire 和 release 的语义。
+ std::memory_order_seq_cst：最强的内存顺序，确保操作的全局顺序。

示例：

```cpp
#include <atomic>
#include <iostream>
#include <thread>

std::atomic<int> counter(0);

void increment() {
    counter.fetch_add(1, std::memory_order_relaxed);  // 原子加 1
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Counter: " << counter.load(std::memory_order_relaxed) << std::endl;  // 原子读取
    return 0;
}
```

## 4. 使用 std::atomic 的注意事项

性能开销：原子操作会引入额外的性能开销，因此应谨慎使用。
内存顺序：选择合适的内存顺序可以减少性能开销，但需要理解其语义。
避免过度使用：原子操作主要用于需要线程安全的场景，避免在不需要的地方使用。

## 5. 总结

std::atomic 是 C++ 中一个非常强大的工具，用于提供原子操作，确保在多线程环境中对变量的访问和修改是线程安全的。
通过合理使用 std::atomic，可以简化线程同步代码，提高代码的可读性和安全性。

## 6. 用下面三个例子进行说明

以下例子以100个线程一起执行，并且同时将全局变量 cnt 取出來 +1 计数，
但是由于多个执行线程同时执行存取 cnt 的关系会造成数据不正确。
来看看結果輸出会是怎么样吧！

```cpp
// g++ std-atomic.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>

using namespace std;

long cnt = 0;

void counter()
{
    for (int i = 0; i < 100000; i++) {
        cnt += 1;
    }
}

int main(int argc, char* argv[])
{
    std::thread threads[100];
    for (int i = 0; i != 100; i++)
    {
        threads[i] = std::thread(counter);
    }
    for (auto &th : threads)
        th.join();
    std::cout << "result: " << cnt << std::endl;
    return 0;
}

// result: 1866806
```

因为 数据从内存读取到寄存器上时， 还存在其他线程也会读取内存中的数据， 导致结果不对

### 添加 mutex 来解决问题

加 mutex 锁来保护临界区域是最常见的做法。可以保证同一时间内只有一个线程会存取 cnt

```cpp
// g++ std-atomic2.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
#include <mutex>

using namespace std;

long cnt = 0;
std::mutex mtx;

void counter()
{
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx);
        //std::cout << std::this_thread::get_id() << ": " << cnt << '\n';
        //mtx.lock();
        cnt += 1;
        //mtx.unlock();
    }
}

int main(int argc, char* argv[])
{
    auto t1 = std::chrono::high_resolution_clock::now();
    std::thread threads[100];
    for (int i = 0; i != 100; i++)
    {
        threads[i] = std::thread(counter);
    }
    for (auto &th : threads)
        th.join();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t2 - t1;
    std::cout << "result: " << cnt << std::endl;
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    return 0;
}

// 输出：
// result: 10000000
// duration: 1426.77 ms
```

输出是正确的， 但是耗时比较长

### 使用 atomic 達到同樣效果，時間約少了7倍

接下來這裡介紹本篇重頭戲 atomic，
如果對象是 long 的話，可以用 `std::atomic<long>`，也可以用 std::atomic_long這個類別，
用 atomic 也可以達到同樣的效果，但所花費的時間有減少嗎？
來看看結果輸出會是怎樣吧！

```cpp
// g++ std-atomic3.cpp -o a.out -std=c++11 -pthread
#include <iostream>
#include <thread>
#include <atomic>

using namespace std;

//std::atomic<long> cnt(0);
std::atomic_long cnt(0);

void counter()
{
    for (int i = 0; i < 100000; i++) {
        cnt += 1;
    }
}

int main(int argc, char* argv[])
{
    auto t1 = std::chrono::high_resolution_clock::now();
    std::thread threads[100];
    for (int i = 0; i != 100; i++)
    {
        threads[i] = std::thread(counter);
    }
    for (auto &th : threads)
        th.join();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t2 - t1;
    std::cout << "result: " << cnt << std::endl;
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    return 0;
}

// 运行结果：
// result: 10000000
// duration: 225.587 ms
```
