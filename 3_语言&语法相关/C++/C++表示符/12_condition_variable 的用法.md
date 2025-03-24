# condition_variable 的用法

std::condition_variable 是 C++11 引入的一个同步原语，用于在多线程环境中实现线程间的等待和通知机制。
它通常与互斥锁（std::mutex）配合使用，允许一个或多个线程在某个条件满足时继续执行。

## 1. std::condition_variable 的基本用法

std::condition_variable 的主要功能包括：

+ 等待条件：线程可以等待某个条件满足。
+ 通知条件：线程可以通知其他线程条件已经满足。

## 2. 主要成员函数

+ wait：使线程进入等待状态，直到被通知或条件满足。
+ wait_for: 使线程进入等待状态，直到被通知或超时。
+ wait_until: 使线程进入等待状态，直到被通知或指定时间点。
+ notify_one：通知一个等待的线程。
+ notify_all：通知所有等待的线程。

## 3. 示例代码

### 3.1 生产者-消费者问题

这是一个经典的多线程问题，生产者线程生成数据，消费者线程消费数据。

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>

std::queue<int> data_queue;
std::mutex mtx;
std::condition_variable cv;
bool done = false;

void producer() {
    for (int i = 0; i < 10; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        data_queue.push(i);
        lock.unlock();
        cv.notify_one();  // 通知一个消费者
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 模拟生产时间
    }

    {
        std::unique_lock<std::mutex> lock(mtx);
        done = true;
    }
    cv.notify_all();  // 通知所有消费者
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return done || !data_queue.empty(); });  // 等待条件变量或队列非空

        if (done && data_queue.empty()) {
            break;  // 如果 done 为 true 且队列为空，退出循环
        }

        int data = data_queue.front();
        data_queue.pop();
        lock.unlock();
        std::cout << "Consumer: " << data << std::endl;
    }
}

int main() {
    std::thread producer_thread(producer);
    std::thread consumer_thread(consumer);

    producer_thread.join();
    consumer_thread.join();

    return 0;
}
```

### 3.2 等待多个条件

有时候需要等待多个条件满足，可以使用逻辑表达式组合条件。

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
int value = 0;

void wait_for_value() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return value > 5 && value < 10; });  // 等待 value 在 6 到 9 之间
    std::cout << "Value: " << value << std::endl;
}

void set_value() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    {
        std::lock_guard<std::mutex> lock(mtx);
        value = 7;
    }
    cv.notify_one();
}

int main() {
    std::thread t1(wait_for_value);
    std::thread t2(set_value);

    t1.join();
    t2.join();

    return 0;
}
```

## 4. 使用 std::condition_variable 的注意事项

必须与互斥锁配合使用：std::condition_variable 需要与 std::mutex 配合使用，以确保线程安全。

+ 避免虚假唤醒：线程可能会在没有通知的情况下被唤醒（虚假唤醒）。因此，wait 函数通常需要一个条件表达式来确保线程只在条件真正满足时继续执行。
+ 通知所有线程：如果多个线程可能在等待同一个条件，可以使用 notify_all 而不是 notify_one。

线程安全：确保在修改共享变量时使用互斥锁，以避免数据竞争。

## 5. 总结

std::condition_variable 是一个强大的同步原语，用于在多线程环境中实现线程间的等待和通知机制。通过合理使用 std::condition_variable，可以实现高效的线程同步，避免线程忙等，提高程序的性能和可维护性。
