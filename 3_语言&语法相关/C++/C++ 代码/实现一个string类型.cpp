#include <iostream>
#include <cstring>
#include <algorithm>

class String {
public:
    String() : data_(nullptr), size_(0) {}

    // 从C字符串构造函数
    String(const char* str) {
        if (str) {
            size_ = std::strlen(str);
            data_ = new char[size_ + 1];
            std::strcpy(data_, str);
        } else {
            data_ = nullptr;
            size_ = 0;
        }
    }

    // 拷贝构造函数
    String(const String& other) {
        size_ = other.size_;
        data_ = new char[size_ + 1];
        std::strcpy(data_, other.data_);
    }

    // 移动构造函数
    String(String&& other) noexcept {
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // 赋值运算符
    String& operator=(const String& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new char[size_ + 1];
            std::strcpy(data_, other.data_);
        }
        return *this;
    }

    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~String() {
        delete[] data_;
    }

    const char* c_str() const {
        return data_;
    }

    size_t size() const {
        return size_;
    }

    // 字符串拼接
    String& operator+=(const String& other) {
        if (other.size_ > 0) {
            char* new_data = new char[size_ + other.size_ + 1];
            std::strcpy(new_data, data_);
            std::strcat(new_data, other.data_);
            delete[] data_;
            data_ = new_data;
            size_ += other.size_;
        }
        return *this;
    }

    // 字符串比较
    bool operator==(const String& other) const {
        return std::strcmp(data_, other.data_) == 0;
    }

    bool operator!=(const String& other) const {
        return !(*this == other)
    }

    // 下标访问
    char& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    const char& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

private:
    char* data_;
    size_t size_;
};

// 字符串拼接
String operator+(const String& lhs, const String& rhs) {
    String rusult = lhs;
    result += rhs;
    return result;
}

// 字符串流运算符重载
std::ostream& operator<<(std::ostream& os, const String& str) {
    os << str.c_str();
    return os;
}

// 示例用法
int main() {
    String str1("Hello");
    String str2("World");
    String str3 = str1 + " " + str2;

    std::cout << str3 << std::endl; // 输出: Hello World

    str3[0] = 'J';
    std::cout << str3 << std::endl; // 输出: Jello World

    return 0;
}
