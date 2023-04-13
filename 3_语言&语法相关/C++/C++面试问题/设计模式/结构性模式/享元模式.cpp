// 享元模式（Flyweight Pattern）主要用于减少创建对象的数量，以减少内存占用和提高性能。
// 这种类型的设计模式属于结构型模式，它提供了减少对象数量从而改善应用所需的对象结构的方式。

// 享元模式尝试重用现有的同类对象，如果未找到匹配的对象，则创建新对象。我们将通过创建 5 个对象来画
// 出 20 个分布于不同位置的圆来演示这种模式。由于只有 5 种可用的颜色，所以 color 属性被用来检查
// 现有的 Circle 对象。

#include <iostream>
#include <string>
#include <map>
#include <random>

using namespace std;

class Shape {
public:
    virtual ~Shape() {}
    virtual void draw() = 0;
};

class Circle : public Shape {
public:
    Circle(string color) : color_(color) {}
    void set_x(int x) {
        x_ = x;
    } 
    void set_y(int y) {
        y_ = y;
    }
    void set_radius(int r) {
        radius_ = r;
    }

    virtual void draw() override {
        cout << "画一个以(" << x_ << ", " << y_ << ")为圆心，" << 
            radius_ << "为半径的" << color_ << "的圆" << endl;
    }
private:
    string color_;
    int x_;
    int y_;
    int radius_;
};

class Factory {
public:
    ~Factory() {
        for(auto iter = circles.begin(); iter != circles.end(); iter++) {

        }
    }
    Circle* get_circle(string color) {
        if(circles.count(color) == 0) {
            circles[color] = new Circle(color);
        }
        return circles[color];
    }
private:
    map<string, Circle*> circles;
};

int main() {
    string colors[] = {"red", "green", "yellow", "blue", "white"};
    Factory* factory = new Factory();
    for (int i = 0; i < 20; i++) {
        int num = random() % 5;
        Circle* circle = factory->get_circle(colors[num]);
        circle->draw();
    }
}