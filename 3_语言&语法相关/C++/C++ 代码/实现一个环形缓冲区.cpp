// https://www.bilibili.com/video/BV1MDXeYHERZ/?spm_id_from=333.1387.upload.video_card.click&vd_source=b259349dd30cfc108bc01ada018c312f

// 实现一个环形缓冲区，要考虑到线程安全

#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <stdexcept>

template <typename T>
class CircleBuffer {
public:
    explicit CircleBuffer(size_t capacity) : 
        capacity_(capacity), buffer_(capacity), readIndex_(0), writeIndex_(0), itemCount_(0) {}
    
    void put(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [this] {
            return itemCount_ < capacity_;
        });
        buffer_[writeIndex_] = item;
        writeIndex_ = (writeIndex_ + 1) % capacity;
        itemCount_++;
        lock.unlock();
        notEmpty_.notify_one();
    }

    // 如果要区分右值，可以添加下面这个函数
    void put(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [this] {
            return itemCount_ < capacity_;
        });
        buffer_[writeIndex_] = std::move(item);
        writeIndex_ = (writeIndex_ + 1) % capacity_;
        itemCount_++;
        lock.unlock();
        notEmpty_.notify_one();
    }

    T get() {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait(lock, [this] {
            return itemCount_ > 0;
        });

        T item = buffer_[readIndex_];
        readIndex_ = (readIndex_ + 1) % capacity_;
        itemCount_--;
        lock.unlock();
        notFull_.notify_one();
        return item;
    }

private:
    size_t capacity_;
    std::vector<T> buffer_;
    size_t readIndex_;
    size_t writeIndex_;
    size_t itemCount_;
    std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
};

int main() {
    CircleBuffer<int> buffer(5);

    std::thread producer([&buffer] {
        for (int i = 1; i < 10; i++) {
            buffer.put(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    std::thread consumer([&buffer] {
        for (int i = 1; i < 10; i++) {
            int item = buffer.get();
            std::cout << "Consumed: " << item << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
    });

    producer.join();
    consumer.join();
    return 0;
}