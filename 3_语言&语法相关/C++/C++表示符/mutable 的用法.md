mutable 的用法
## mutable 在类中的使用
mutable 只能用来修饰类的数据成员，而被 mutable 修饰的数据成员， 可以在 const 成员函数中修改
```cpp
class HashTable {
public:
    std::stringf lookup(const std::string& key) const {
        if (key == last_key_) return last_value_;
        std::string value {this->lookupInternal(key)};
        last_key_ = key;
        last_value_ = value;
        return value;
    }
private:
    mutable std::string last_key_;
    mutable std::string last_value_;
};
```
## mutable 在表达式 lambda 中的使用
C++11 引入了 lambda 表达式， 程序员可以凭此创建匿名函数。 在 Lambda 表达式的设计中， 捕获变量的几种方式， 其中按值捕获的方式不允许程序员在 lambda 函数的 函数体中修改捕获的变量。 而以 `mutable` 修饰 lambda 函数， 则可以打破这种限制
```cpp
int x = 0;
auto f1 = [=]() mutable {x = 42;} // 正确， 创建了一个函数类型的实例
auto f2 = [=]() {x = 42;} //错误， 不允许修改按值捕获的外部变量的值
```