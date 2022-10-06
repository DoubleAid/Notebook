关键字 constexpr 是C++11中引入的关键字，声明为constexpr类型的变量，编译器会验证该变量的值是否是一个常量表达式。
声明为constexpr的变量一定是一个常量，而且必须用常量表达式初始化：

```cpp
constexpr int mf = 0; // 0 是 常量表达式
constexpr int limit = mf + 2; // mf + 1 是常量表达式
constexpr int sz = size() // 只有当 size() 是一个 返回值为 constexpr 函数时才是一个正确的声明语句
```