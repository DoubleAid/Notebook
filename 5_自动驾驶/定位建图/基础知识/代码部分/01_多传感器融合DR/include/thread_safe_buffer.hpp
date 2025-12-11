// 工具类：线程安全缓存

#pragma once

#include <deque>
#include <mutex>
#include <memory>

template <typename T>
class ThreadSafeBuffer {
public:
    ThreadSafeBuffer() : max_size_(100) {}
    ~ThreadSafeBuffer() = default;

    // 压入数据（超出最大长度则弹出最旧数据）
    void push(const T& data) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (buffer_.size() > max_size_) {
            buffer_.pop_front();
        }
        buffer_.push_back(data);
    }

    // 弹出最前数据
    bool pop_front(T& data) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (buffer_.empty()) {
            return false;
        }
        data = buffer_.front();
        buffer_.pop_front();
        return true;
    }

    // 获取最后两个数据（用于激光帧间配准）
    bool get_last_two(T& prev, T& curr) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (buffer_.size() < 2) {
            return false;
        }
        prev = buffer_[buffer_.size() - 2];
        curr = buffer_.back();
        return true;
    }

    // 
    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        buffer_.clear();
    }

private:
    std::deque<T> buffer_;
    std::mutex mtx_;
    const size_t max_size_;
};