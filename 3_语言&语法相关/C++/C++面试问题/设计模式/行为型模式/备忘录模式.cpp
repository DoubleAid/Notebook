// 备忘录模式（Memento Pattern）保存一个对象的某个状态，以便在适当的时候恢复对象。备忘录模式属于行为型模式。
// 意图：在不破坏封装性的前提下，捕获一个对象的内部状态，并在该对象之外保存这个状态。
// 主要解决：所谓备忘录模式就是在不破坏封装的前提下，捕获一个对象的内部状态，并在该对象之外保存这个状态，
// 这样可以在以后将对象恢复到原先保存的状态。
// 何时使用：很多时候我们总是需要记录一个对象的内部状态，这样做的目的就是为了允许用户取消不确定或者错误的操作，
// 能够恢复到他原先的状态，使得他有"后悔药"可吃。
// 如何解决：通过一个备忘录类专门存储对象状态。

// 关键代码：客户不与备忘录类耦合，与备忘录管理类耦合。
// 应用实例： 1、后悔药。 2、打游戏时的存档。 3、Windows 里的 ctrl + z。 4、IE 中的后退。 5、数据库的事务管理。
// 优点： 1、给用户提供了一种可以恢复状态的机制，可以使用户能够比较方便地回到某个历史的状态。 2、实现了信息的封装，使得用户不需要关心状态的保存细节。
// 缺点：消耗资源。如果类的成员变量过多，势必会占用比较大的资源，而且每一次保存都会消耗一定的内存。
// 使用场景： 1、需要保存/恢复数据的相关状态场景。 2、提供一个可回滚的操作。
// 注意事项： 1、为了符合迪米特原则，还要增加一个管理备忘录的类。 
//          2、为了节约内存，可使用原型模式+备忘录模式。

#include <iostream>

using namespace std;

class SnapShot {
public:
    SnapShot(string text, int x, int y, double width)
        : text_(text), cur_x_(x), cur_y_(y), selection_width_(width) {}
    string get_text() {
        return text_;
    }
    int get_cur_x() {
        return cur_x_;
    }
    int get_cur_y() {
        return cur_y_;
    }
    double get_selection_width() {
        return selection_width_;
    }
private:
    const string text_;
    const int cur_x_;
    const int cur_y_;
    const double selection_width_;
};

class Editor {
public:
    void setText(string text) { text_ = text; }
    void setCursor(int x, int y) {
        cur_x_ = x;
        cur_y_ = y;
    }
    void setSelectionWidth(double width) {
        selection_width_ = width;
    }
    shared_ptr<SnapShot> createSnapShot() {
        auto res = make_shared<SnapShot>(text_, cur_x_, cur_y_, selection_width_);
        cout << "创建编辑器快照完成" << endl;
        return res;
    }
    void restore(shared_ptr<SnapShot> ss_) {
        text_ = ss_->get_text();
        cur_x_ = ss_->get_cur_x();
        cur_y_ = ss_->get_cur_y();
        selection_width_ = ss_->get_selection_width();
        cout << "恢复编辑器状态成功" << endl;
    }
private:
    string text_;
    int cur_x_;
    int cur_y_;
    double selection_width_;
};