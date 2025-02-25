#include <vector>
#include <queue>
#include <mutex>
#include <algorithm>
#include <future>
#include <functional>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include <thread>

class ThreadPool {
public:
    ThreadPool(const int thread_count);
    ThreadPool() = delete;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool operator=(const ThreadPool&) = delete;

    template<typename Func, typename... Args>
    auto enqueue(Func&& func, Args&&... args)
    -> std::future<typename std::invoke_result<Func, Args...>::type>;
    
    ~ThreadPool();

    void wait(uint64_t interval = 500);

    void clear();

private:
    using Job = std::function<void()>;
    std::queue<Job> jobs_;

    static void threadTask(ThreadPool* pool);

    std::vector<std::thread> threads_;

    std::atomic<int> jobs_count_;
    std::atomic<bool> terminate_;

    mutable std::mutex jobs_mutex_;

    std::condition_variable jobs_available_;
};

ThreadPool::ThreadPool(const int thread_count) : jobs_count_(0), terminate_(false) {
    threads_.reserve(thread_count);
    std::generate_n(std::back_inserter(threads_), thread_count, [this]() {
        return std::thread(threadTask, this);
    });
}

template<typename Func, typename... Args>
auto ThreadPool::enqueue(Func&& func, Args&&... args) -> std::future<typename std::invoke_result<Func, Args...>::type> {
    using Task = std::packaged_task<typename std::invoke_result<Func, Args...>::type()>;
    auto task = std::make_shared<Task>(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));
    {
        std::lock_guard<std::mutex> lock{jobs_mutex_};
        jobs_.emplace([task]() {
            (*task)();
        });
    }
    jobs_count_.fetch_add(1);
    jobs_available_.notify_one();
    return task->get_future();
}

void ThreadPool::wait(uint64_t interval) {
    while (jobs_count_.load() > 0 || !terminate_) {
        std::this_thread::sleep_for(std::chrono::microseconds(interval));
    }
}

void ThreadPool::clear() {
    std::lock_guard<std::mutex> lock{jobs_mutex_};
    while (!jobs_.empty()) {
        jobs_.pop();
    }
}

ThreadPool::~ThreadPool() {
    clear();
    terminate_ = true;
    jobs_available_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::threadTask(ThreadPool* pool) {
    while (!pool->terminate_) {
        std::cout << "try to get a job" << std::endl;
        std::unique_lock<std::mutex> lock{pool->jobs_mutex_};
        pool->jobs_available_.wait(lock, [pool] {
            return (!pool->jobs_.empty()) || pool->terminate_;
        });
        if (pool->terminate_) break;
        pool->jobs_count_.fetch_sub(1);
        auto job = std::move(pool->jobs_.front());
        pool->jobs_.pop();
        job();
    }
}

int add(int a, int b) {
    std::cout << "a + b = " << a + b << std::endl;
    return a + b;
}

int mul(int a, int b) {
    std::cout << "a * b = " << a * b << std::endl;
    return a * b;
}

int main() {
    ThreadPool pool(4);
    for (size_t i = 0; i < 100; i++) {
        if (i % 2 == 0) {
            pool.enqueue(add, 5, i);
        } else {
            pool.enqueue(mul, 5, i);
        }
    }
    pool.wait();
    return 0;
}