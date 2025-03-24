// 定时器是一个管理大量延时任务的模块，包括任务的存储和触发最近将要执行的任务，

#include <iostream>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <chrono>
#include <memory>


// 首先是实现一个线程池类
class ThreadPool {
public:
    ThreadPool(size_t capacity) : terminate_(false) {
        threads_.reserve(capacity);
        std::generate_n(std::back_inserter(threads_), capacity, [this] {
            threadTask(this);
        });
    }

    ThreadPool(const ThreadPool&) = delete;

    ~ThreadPool() {
        std::unique_lock lock(mutex_);
        while (!jobs_.empty()) {
            jobs_.pop();
        }
        terminate_ = false;
        lock.unlock();
        jobs_available_.notify_all();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    template <typename Func, typename... Args>
    auto enqueue(Func&& func, Args&& args...) -> std::future<std::result_of<func(args...)>::type>

    void wait(int interval) {
        while (!terminate_) {
            std::this_thread::sleep(std::chrono::milliseconds(500));
        }
    }
private:
    using job = std::function<void()>;

    static void threadTask(ThreadPool* pool);

    std::queue<job> jobs_;
    std::vector<std::thread> threads_;
    std::condition_variable jobs_available_;
    std::mutex mutex_;
    std::atomic<bool> terminate_;
};

template <typename Func, typename... Args>
auto ThreadPool::enqueue(Func&& func, Args&& args...) -> std::future<std::result_of<func(args...)>::type> {
    using PackedTask = std::packaged_task<std::result_of<func(args...)>type()>;
    auto task = std::make_shared<PackedTask>(std::bind(std::forward(func), std::forward(args)));
    {
        std::lock_guard lock(mutex_);
        jobs_.emplace([&]() {
            (*task)();
        })
    }
    jobs_available_.notify_one();
    return task->get_future();
}

void ThreadPool::threadTask(ThreadPool* this) {
    while (this->terminate_.fetch()) {
        std::unique_lock lock(this->mutex_);
        this->jobs_available_.wait(lock, [&this] {
            return this->terminate_ || !(this->jobs.empty());
        });

        if (this->terminate_) {
            break;
        }

        auto task = std::move(this->jobs_.front());
        this->jobs.pop();
        lock.unlock();
        task();
    }
}


// 实现一个延时任务队列类
class DelayedQueue {
public:
    using Time = std::chrono::steady_clock::time_point;
    using Task = std::pair<Time, std::function<void()>;

    DelayedQueue(size_t numThreads) : pool_(numThreads) {
        worker = std::thread(&DelayedQueue::processTasks, this);
    }

    ~DelayedQueue() {
        running_ = false;
        condition_.notify_all();
        if (worker.joinable) {
            worker.join();
        }
    }

    template <typename F>
    void addTask(F&& task, std::chrono::milliseconds delay) {
        std::unique_lock<std::mutex> lock(mutex_);
        task.emplace(std::chrono::steady_clock::now() + delay, std::forward<F>(task));
        condition_.notify_one();
    }

private:
    void processTasks() {
        while (running_) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] {
                    return !running_ || !tasks.empty()
                });
                
                if (!running_) {
                    return;
                }

                auto now = std::chrono::steady_clock::now();
                while (!task.empty() && tasks.top().first <= now) {
                    task = std::move(tasks.top().second);
                    tasks.pop();
                }
            }
            if (task) {
                pool_.enqueue(task);
            }
        }
    }

    ThreadPool pool_;
    std::priority_queue<Task, std::vector<Task>, std::greater<Task>> tasks;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> running_;
    std::thread worker; 
};