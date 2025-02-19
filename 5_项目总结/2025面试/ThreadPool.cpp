#include <iostream>
#include <atomic>
#include <vector>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <algorithm>
#include <functional>
#include <thread>
#include <future>

using namespace std;

class ThreadPool {
public:
    ThreadPool(size_t thread_count);
    ThreadPool() = delete;
    ~ThreadPool();
    ThreadPool operator=(const ThreadPool&) = delete;
    ThreadPool(const ThreadPool&) = delete;

    template<typename Func, typename... Args>
    auto enqueue(Func&& func, Args&&... args) -> std::future<typename std::invoke_result_t<Func, Args...>>;

    void clear();
    void wait(uint64_t interval = 500);

private:
    using Job = std::function<void()>;
    std::queue<Job> jobs_;
    std::vector<std::thread> threads_;

    std::condition_variable available_jobs_;
    std::mutex job_mutex_;

    static void threadTask(ThreadPool* pool);
    std::atomic<int> jobs_count_;
    std::atomic<bool> terminate_;
};

ThreadPool::ThreadPool(size_t thread_count) : jobs_count_(0), terminate_(false) {
    threads_.reserve(thread_count);
    std::generate_n(std::back_inserter(threads_), thread_count, [this]() {
        return std::thread(threadTask, this);
    });
}

ThreadPool::~ThreadPool() {
    clear();
    terminate_ = true;
    available_jobs_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

template<typename Func, typename... Args>
auto ThreadPool::enqueue(Func&& func, Args&&... args) -> std::future<typename std::invoke_result_t<Func, Args...>> {
    using PackedTask = std::packaged_task<typename std::invoke_result_t<Func, Args...>()>;

    jobs_count_.fetch_add(1);
    auto task = std::make_shared<PackedTask>(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));

    auto ret = task->get_future();
    {
        std::lock_guard<std::mutex> lock{job_mutex_};

        jobs_.emplace([task]() { (*task)(); });
    }
    available_jobs_.notify_one();
    return ret;
}

void ThreadPool::clear() {
    std::lock_guard<std::mutex> lock{job_mutex_};
    while (!jobs_.empty()) jobs_.pop();
}

void ThreadPool::wait(uint64_t interval) {
    while (jobs_count_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }
}

void ThreadPool::threadTask(ThreadPool* pool) {
    while (!pool->terminate_) {
        std::unique_lock<std::mutex> lock{pool->job_mutex_};
        if (pool->jobs_.empty()) {
            pool->available_jobs_.wait(lock, [&]() { return pool->terminate_ || !(pool->jobs_.empty()); });
        }

        if (pool->terminate_) break;

        auto job = std::move(pool->jobs_.front());
        pool->jobs_.pop();
        lock.unlock();

        job();

        pool->jobs_count_.fetch_sub(1);
    }
}

void sleepTask() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

std::mutex func_mutex;

void add(int a, int b, int thread_id) {
    sleepTask();
    std::lock_guard<std::mutex> lock{func_mutex};
    std::cout << "thread id = " << thread_id << ", a + b = " << a + b << std::endl;
}

void mul(int a, int b, int thread_id) {
    sleepTask();
    std::lock_guard<std::mutex> lock{func_mutex};
    std::cout << "thread id = " << thread_id << ", a * b = " << a * b << std::endl;
}

int main() {
    ThreadPool pool(5);
    for (size_t i = 0; i < 100; i++) {
        if (i % 2 == 0) {
            pool.enqueue(add, 5, 7, i);
        } else {
            pool.enqueue(mul, 5, 7, i);
        }
    }
    pool.wait();
    return 0;
}