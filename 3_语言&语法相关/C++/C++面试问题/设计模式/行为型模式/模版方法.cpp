// 早绑定： App 调用 library 的方法， 主程序在 APP 内
// 晚绑定： App 继承 library 的类， 主程序在 Library 内
// 模版模式就是使用晚绑定替换早绑定
#include <iostream>
using namespace std;

class Library {
public:
virtual ~Library() = 0;
virtual void A_func() = 0;
virtual void B_func() = 0;
virtual void C_func() = 0;
void process() {
    A_func();
    B_func();
    C_func();
}
};

class App : Library {
public:
virtual void A_func() override {
    cout << "A func" << endl;
}
virtual void B_func() override {
    cout << "B func" << endl;
}
virtual void C_func() override {
    cout << "C func" << endl;
}
};