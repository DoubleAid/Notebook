// 组合模式（Composite Pattern），又叫部分整体模式，是用于把一组相似的对象当作一个单一的对象。
// 组合模式依据树形结构来组合对象，用来表示部分以及整体层次。这种类型的设计模式属于结构型模式，
// 它创建了对象组的树形结构。

// 这种模式创建了一个包含自己对象组的类。该类提供了修改相同对象组的方式。
// 我们通过下面的实例来演示组合模式的用法。实例演示了一个组织中员工的层次结构。

// 意图：将对象组合成树形结构以表示"部分-整体"的层次结构。组合模式使得用户对单个对象和组合对象的使用具有一致性。

// 主要解决：它在我们树型结构的问题中，模糊了简单元素和复杂元素的概念，
// 客户程序可以像处理简单元素一样来处理复杂元素，从而使得客户程序与复杂元素的内部结构解耦。
// 何时使用： 1、您想表示对象的部分-整体层次结构（树形结构）。 
//          2、您希望用户忽略组合对象与单个对象的不同，用户将统一地使用组合结构中的所有对象。

// 如何解决：树枝和叶子实现统一接口，树枝内部组合该接口。
// 关键代码：树枝内部组合该接口，并且含有内部属性 List，里面放 Component。
// 应用实例： 1、算术表达式包括操作数、操作符和另一个操作数，其中，另一个操作数也可以是操作数、操作符和另一个操作数。 
// 2、在 JAVA AWT 和 SWING 中，对于 Button 和 Checkbox 是树叶，Container 是树枝。
// 优点： 1、高层模块调用简单。 2、节点自由增加。

// 缺点：在使用组合模式时，其叶子和树枝的声明都是实现类，而不是接口，违反了依赖倒置原则。
// 使用场景：部分、整体场景，如树形菜单，文件、文件夹的管理。
// 注意事项：定义时为具体类。
#include <iostream>
#include <string>
#include <map>

using namespace std;

class Graphic {
public:
    virtual ~Graphic() {}
    virtual void move2somewhere(int x, int y) = 0;
    virtual void draw() = 0;
};

class Dot : public Graphic {
public:
    Dot(int x, int y) : x_(x), y_(y) {}
    void move2somewhere(int x, int y) override {
        x_ += x;
        y_ += y;
    }
    void draw() override {
        cout << "在("<< x_ << ", " << y_ << ")绘制一个点" << endl;
    }
private:
    int x_;
    int y_;
};

class Circle : public Graphic {
public:
    Circle(int x, int y, int r) : x_(x), y_(y), radius_(r){}
    void move2somewhere(int x, int y) override {
        x_ += x;
        y_ += y;
    }
    void draw() override {
        cout << "以("<< x_ << ", " << y_ << ")为圆心，以"<< radius_<<"半径绘制一个圆" << endl;
    }
private:
    int x_;
    int y_;
    int radius_;
};

class ComposeGraphic : public Graphic {
public:
    void add(int id, Graphic* child) {
        children_[id] = child;
    }
    void erase(int id) {
        children_.erase(id);
    }
    void move2somewhere(int x, int y) override {
        for(auto iter = children_.cbegin(); iter != children_.cend(); iter++) {
            iter ->second->move2somewhere(x, y);
        }
    }
    void draw() override {
        for (auto iter = children_.cbegin(); iter != children_.cend(); iter++) {
            iter->second->draw();
        }
    }
private:
    map<int, Graphic*> children_;
};

int main() {
    ComposeGraphic* all = new ComposeGraphic();
    Dot* dot1 = new Dot(1, 2);
    Circle* cir1 = new Circle(5, 2, 2);
    ComposeGraphic* child_graph = new ComposeGraphic();
    Dot* dot2 = new Dot(4, 7);
    Dot* dot3 = new Dot(7, 8);
    child_graph->add(0, dot2);
    child_graph->add(1, dot3);
    
    all->add(0, dot1);
    all->add(1, cir1);
    all->add(2, child_graph);

    all->draw();

    delete all;
    delete dot1;
    delete dot2;
    delete dot3;
    delete cir1;

    return 0;
}