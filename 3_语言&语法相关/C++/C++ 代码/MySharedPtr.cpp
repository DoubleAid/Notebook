#include <atomic>
#include <utility>

template <typename T>
class MySharedPtr {
private:
    T* ptr;
    std::atomic<int>* count;

    void release() {
        if (ptr && --(*count) == 0) {
            delete ptr;
            delete count;
        }
    }

public:
    // 构造函数
    explicit MySharedPtr(T* p = nullptr): ptr(p), count(new std::atomic<int>(1)) {
        if (!ptr) {
            delete count;
            count = nullptr;
        }
    }

    // 拷贝构造函数
    MySharedPtr(const MySharedPtr& other) : ptr(other.ptr), count(other.count) {
        if (ptr) {
            (*count)++;
        }
    }

    // 移动构造函数
    MySharedPtr(MySharedPtr&& other) noexcept : ptr(other.ptr), count(other.count) {
        other.ptr = nullptr;
        other.count = nullptr;
    }

    // 赋值运算符
    MySharedPtr& operator=(const MySharedPtr& other) noexcept {
        if (this != &other) {
            // 释放当前资源
            release();
            ptr = other.ptr;
            count = other.count;
            if (ptr) {
                ++(*count);
            }
        }
        return *this;
    }

    // 移动赋值运算符
    MySharedPtr& operator=(MySharedPtr&& other) noexcept {
        if (this != &other) {
            // 释放当前资源
            release();
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = nullptr;
        }
        return *this;
    }

    ~MySharedPtr() {
        release();
    }

    T& operator*() const {
        return *ptr;
    }

    T* operator->() {
        return ptr;
    }

    T* get() const {
        return ptr;
    }

    long use_count() const {
        return count ? *count : 0;
    }
};

int main() {
    MySharedPtr<int> sp1(new int(10));
    std::cout << "sp1 use count: " << sp1.use_count() << std::endl;
    {
        MySharedPtr<int> sp2 = sp1; // 拷贝构造
        std::cout << "sp1 use count: " << sp1.use_count() << std::endl;
        std::cout << "sp2 use count: " << sp2.use_count() << std::endl;
    }
    std::cout << "sp1 use count: " << sp1.use_count() << std::endl;

    return 0;
}