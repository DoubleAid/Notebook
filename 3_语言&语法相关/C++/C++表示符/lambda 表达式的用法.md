# lambda表达式的用法

## [参考链接](jianshu.com/p/a200a2dab960)

## 什么是 Lambda 表达式

`lambda` 表达式是 `c++` 提出的新特性， 主要是用来实现代码中 函数的内嵌； 简化了编程， 提高了效率
它的基本格式如下：

```cpp
auto fun = [捕获参数](函数参数) {函数体};
```

## 基本语法

### 最简单的 lambda

```cpp
int main(void) {
    [](){cout << "hello world" << endl;};
    return 0;
}
```

### lambda 可以赋值、传递参数和返回值

```cpp
int main(void) {
    int num = 100;
    auto fun = [](int num) { num = 5; cout << num << endl; return 0;};
    fun(num);
    cout << num << endl;
    return 0;
}
```

### []的使用

lambda 表达式的 [] 用来确定捕获参数

+ [=] : 捕获的局部变量只可读不可写， 捕获范围是当前 lambda 表达式之前的作用域
+ [&] : 捕获的局部变量可读可写

```cpp
int main(void) {
    int num = 100;
    auto fun1 = [=]() {cout << num << endl;};
    fun1();

    auto fun2 = [&num]() {num = 200; cout << num << endl;};
    fun2();

    cout << num << endl;
    return 0;
}
```

#### lambda 没有地址

lambda 表达式是内联展开的， 没有实际的地址， 这是和普通函数的一个很大的区别

#### lambda 在 class 中的使用

lambda 在 c++ class 中的使用需要知道如何捕获 this

```cpp
void MyClass::function() {
    // 使用 this 来捕获 this
    auto fun1 = [this](int v){cout << v+this->num << endl;};
    // 使用 [&] 捕获所有父作用域的引用， 包括 this
    auto fun2 = [&](int v){cout << v + this->num << endl;};
}
```
