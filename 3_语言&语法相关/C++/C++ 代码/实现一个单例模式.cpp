#include <iostream>

using namespace std;

template <typename T>
class Singleton {
public:
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    // 饿汉式，即在函数执行时就进行初始化
    static T& get_instance() {
        static T instance;
        return instance;
    }

    // 懒汉式，即只有在使用时才进行初始化
    static shared_ptr<T> get_instance() {
        if (!instance_) {
            std::lock_guard lock(mutex_);
            if (!instance_) instance_ = new T();
        }
        return instance_;
    }

private:
    Singleton() = default;
    ~Singleton() = default;

    // 懒汉式需要添加一个锁和一个实例来保证线程安全
    static shared_ptr<T> instance_ = nullptr;
    std::mutex mutex_;
};

