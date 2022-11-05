类中的默认的成员函数：
+ 默认构造函数
+ 默认析构函数
+ 拷贝构造函数
+ 拷贝赋值函数
+ 移动构造函数
+ 移动拷贝函数

C++规定， 一旦程序实现了这些函数的自定义版本， 则编译器不会再自动生产默认版本，注意只是不自动生成默认版本， 但可以手动生成默认版本的。 当我们自己定义了 带参数的构造函数时， 我们最好声明不带参数的版本完成无参的变量初始化；此时编译是不会再自动提供默认的无参版本了， 但我们可以通过使用 default 来控制默认构造函数的生成， 显示的指示编译器生成该函数的默认版本， 比如：
```cpp
class MyClass {
  public:
    MyClass() = default;
    MyClass(int i) : data(i) {}
  private:
    int data;
};
```

有的时候希望禁止某些默认函数的生成， 以往的做法是将默认函数声明为 private 并不提供实现， C++开始则使用 delete 关键字显式指示编译器不生成函数的默认版本。
```cpp
class MyClass {
  public:
    MyClass() = default;
    MyClass(const MyClass&) = delete;
}
```