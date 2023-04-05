# explicit 的作用
## 隐式转换
```CPP
#include <iostream>
using namespace std;

class Point {
public:
    int x, y;
    Point(int x = 0, int y = 0) : x(x), y(y) {}
};

void displayPoint(const Point& p) {
    cout << "(" << p.x << "," << p.y << ")" << endl;
}

int main() {
    displayPoint(1); // 1 会使用Point构造函数里面的默认构造函数
    Point p = 1；
}
```
定义的Point类的构造函数使用了默认参数， 主函数中的两行都会出发该构造函数的隐式调用。displayPoint需要Point类型的参数， 传入的 int 型会调用 Point的构造函数

这种隐式发生的构造调用有时可以带来便利， 有时也会有意想不到的后果， explicit 关键字用来避免这种情况发生

## explicit 关键字
在 C++ 中， explicit 关键字用来修饰类的构造函数， 被修饰的构造函数的类， 不会发生相应的隐式类型转换， 只能以显示的方式进行类型转换

explicit 使用注意事项：
+ explicit 关键字作用于类内部的构造函数或者转换函数(C++起)， 指定构造函数或转换函数为显式， 即它不能用于 隐式转换 和 复制初始化
+ explicit 表示符可以与常量表达式一起使用，函数当且仅当该常量表达式求值为 true 才为显式

构造函数被 explicit 修饰后， 就不能再被隐式调用了，之前的代码 在 Point(int x=0, int y=0) 前加上explicit修饰， 就无法便已通过

之后的代码需要修改成以下的形式
```cpp
#include <iostream>
using namespace std;

class Point {
public:
    int x, y;
    explicit Point(int x = 0, int y = 0) : x(x), y(y) {}
};

void displayPoint(const Point& p) {
    cout << "(" << p.x << "," << p.y << ")" << endl;
}

int main() {
    displayPoint(Point(1)); // 1 会使用Point构造函数里面的默认构造函数
    Point p(1);
}
```