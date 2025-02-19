#include <iostream>
#include <atomic>
#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <future>
#include <functional>
#include <algorithm>

class ThreadPool {
public:
    ThreadPool(int thread_count);
    ~ThreadPool();
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(const ThreadPool&) = delete;

    template <typename Func, typename... Args>
    auto enqueue(Func&& func, Args&&... args)
    -> std::future<typename std::result_of<Func(Args...)>::type>;

    std::size_t threadCount() const;
    std::size_t jobsWaiting() const;
    std::size_t jobsTotal() const;

    void clear();
    void wait(int64_t interval = 500);

private:
    static void threadTask(ThreadPool* pool);

    using Job = std::function<void()>;
    std::queue<Job> jobs_;
    mutable std::mutex jobs_mutex_;

    std::condition_variable jobs_available_;
    std::vector<std::thread> threads_;

    std::atomic<int> jobs_count_;
    std::atomic<bool> terminate_;
};

ThreadPool::ThreadPool(int thread_count) : jobs_count_(0), terminate_(false) {
    threads_.reserve(thread_count);
    // generate_n 的第三个参数必须是 可执行函数
    std::generate_n(std::back_inserter(threads_), thread_count, [this]() {
        return std::thread{threadTask, this};
    });
}

ThreadPool::~ThreadPool() {
    clear();
    terminate_ = true;
    jobs_available_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

template<typename Func, typename... Args>
auto ThreadPool::enqueue(Func &&func, Args &&...args) -> std::future<typename std::result_of<Func(Args...)>::type> {
    using PackedTask = std::packaged_task<typename std::result_of<Func(Args...)>::type()>;

    jobs_count_.fetch_add(1);
    auto task = std::make_shared<PackedTask>(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));

    auto ret = task->get_future();
    {
        std::lock_guard<std::mutex> lock{jobs_mutex_};
        jobs_.emplace([task]() { (*task)(); });
    }
    jobs_available_.notify_one();
    return ret;
}

std::size_t ThreadPool::threadCount() const {
    return threads_.size();
}

std::size_t ThreadPool::jobsWaiting() const {
    auto jobs_total = jobs_count_.load();
    return jobs_total > threads_.size() ? jobs_total - threads_.size() : 0;
}

std::size_t ThreadPool::jobsTotal() const { return jobs_count_.load(); }

void ThreadPool::clear() {
    std::lock_guard<std::mutex> lock{jobs_mutex_};
    while (!jobs_.empty()) jobs_.pop();
}

void ThreadPool::wait(int64_t interval) {
    while (jobs_count_.load() != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }
}

void ThreadPool::threadTask(ThreadPool *pool) {
    while (true) {
        if (pool->terminate_) break;
        std::unique_lock<std::mutex> jobs_lock{pool->jobs_mutex_};
        if (pool->jobs_.empty()) {
            pool->jobs_available_.wait(jobs_lock, [&]() { return pool->terminate_ || !(pool->jobs_.empty()); });
        }

        if (pool->terminate_) break;
        auto job = std::move(pool->jobs_.front());
        pool->jobs_.pop();
        jobs_lock.unlock();

        job();

        pool->jobs_count_.fetch_add(-1);
    }
}

void sleepTask() {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
}

std::mutex func_mutex;

int mul(int a, int b, int i) {
    sleepTask();
    std::lock_guard<std::mutex> lock{func_mutex};
    std::cout << "i = " << i << "a = " << a << "  b = " << b << "  a * b = " << a*b << std::endl;
    return 0;
}

int add(int a, int b, int i) {
    sleepTask();
    std::lock_guard<std::mutex> lock{func_mutex};
    std::cout << "i = " << i << "a = " << a << "  b = " << b << "  a + b = " << a+b << std::endl;
    return 0;
}


int main() {
    std::cout << "Hello, World!" << std::endl;
    ThreadPool pool(5);
    for (int i = 0; i < 100; i++) {
        if (i % 2) {
            pool.enqueue(mul, 5, 6, i);
        }
        else {
            pool.enqueue(add, 5, 6, i);
        }
    }
    pool.wait();
    return 0;
}