#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <atomic>
#include <future>
#include <functional>
#include <condition_variable>


class ThreadPool {
public:
    using Ids = std::vector<std::thread::id>;
    ThreadPool(std::size_t thread_pool = std::thread::hardware_concurrency());
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool();
    void clear();
    template <typename Func, typename... Args>
    auto enqueue(Func&& func, Args&&... args)
        -> std::future<typename std::result_of<Func(Args...)>::type>;

private:
    using Job = std::function<void()>;
    static void threadTask(ThreadPool* pool);
    std::queue<Job> jobs_;
    mutable std::mutex jobs_mutex_;
    std::condition_variable jobs_available_;
    std::vector<std::thread> threads_;
    std::atomic<std::size_t> jobs_count_;
    std::atomic<bool> terminate_;
};

ThreadPool::ThreadPool(std::size_t thread_count) : jobs_count_(0), terminate_(false) {
    threads_.reserve(thread_count);
    std::generate_n(std::back_inserter(threads_), thread_count, [this]() {
        return std::thread{threadTask, this};
    });
}

void ThreadPool::clear() {
    std::lock_guard<std::mutex> lock{jobs_mutex_};
    while (!jobs_.empty()) jobs_.pop();
}

ThreadPool::~ThreadPool() {
    clear();
    terminate_ = true;
    jobs_available_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::threadTask(ThreadPool *pool) {
    // keep thread alive
    while(true) {
        if (pool->terminate_) break;
        std::unique_lock<std::mutex> jobs_lock{pool->jobs_mutex_};

        // if there are no more jobs
        if (pool->jobs_.empty()) {
            pool->jobs_available_.wait(
                jobs_lock, [&]() { return pool->terminate_ || !(pool->jobs_.empty()); }
            );
        }

        if (pool->terminate_) break;

        // take next job
        auto job = std::move(pool->jobs_.front());
        pool->jobs_.pop();
        jobs_lock.unlock();
        job();
        pool->jobs_count_.fetch_add(-1);
    }
}

template <typename Func, typename... Args>
auto ThreadPool::enqueue(Func&& func, Args&&... args) 
        -> std::future<typename std::result_of<Func(Args...)>::type> {
    using PackedTask = std::packaged_task<typename std::result_of<Func(Args...)>::type()>;
    jobs_count_.fetch_add(1);
    auto task = std::make_shared<PackedTask>(
        std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
    );
    auto ret = task->get_future();
    {
        std::lock_guard<std::mutex> lock{jobs_mutex_};
        jobs_.emplace([Task]() { (*task)(); });
    }
    jobs_available_.notify_one();
    return ret;
}


// 使用
void func() {};
void main() {
    ThreadPool pool = ThreadPool(5);
    for (int i = 0; i < 6; i++) {
        pool.enqueue(&func);
    }
    pool.wait();
}