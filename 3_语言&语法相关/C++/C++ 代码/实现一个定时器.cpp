// 定时器是一个管理大量延时任务的模块，包括任务的存储和触发最近将要执行的任务，

// 首先是实现一个线程池类

#include <iostream>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

class ThreadPool {
public:

};

// 实现一个延时任务队列类
#include <iostream>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <memory>



class DelayedQueue {
public:
    using Time = std::chrono::steady_clock::time_point;

    DelayedQueue(size_t numThreads);
private:
    void processTasks() {
        while (running_) {
            std::function<void()> task;
        }
    }

    ThreadPool pool_;
    std::priority_queue<std::pair<Time, std::function<void()>>;
    std::vector<std::pair<Time, std::function<void()>>;
}