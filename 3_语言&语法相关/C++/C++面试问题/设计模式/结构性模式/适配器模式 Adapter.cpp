// 考虑经典的“方钉和圆孔”问题， 需要让 方钉能适配圆孔

// 适配器让方针假扮成一个圆钉， 其半径等于方钉对角线的一半，
// 圆钉： 用户端接口
#include <iostream>
#include <string>

using namespace std;

class RoundPeg {
public:
    RoundPeg() {}
    virtual int get_radius() = 0;
};

// 方钉
class SquarePeg {
public:
    explicit SquarePeg(int w) : width_(w) {}
    int get_width() {
        return width_;
    }
private:
    int width_;
};

class SquarePegAdapter : public RoundPeg {
public:
    explicit SquarePegAdapter(SquarePeg* peg) : square_peg_(peg) {} 
    virtual int get_radius() override {
        return square_peg_->get_width();
    }
private:
    SquarePeg* square_peg_;
};

// 圆洞 ： 用户端
class RoundHole {
public:
    explicit RoundHole(int t) : radius_(t) {}
    int get_radius() {
        return radius_;
    }
    bool isFit(RoundPeg* peg) {
        return radius_ >= peg->get_radius();
    }
private:
    int radius_;
};

int main() {
    RoundHole* hole = new RoundHole(10);

    SquarePeg* small_square_peg = new SquarePeg(5);
    SquarePeg* big_square_peg = new SquarePeg(50);
    SquarePegAdapter* small_peg_adapter = new SquarePegAdapter(small_square_peg);
    SquarePegAdapter* big_peg_adapter = new SquarePegAdapter(big_square_peg);

    if(hole->isFit(small_peg_adapter)) {
        cout << "small fit" << endl;
    } else {
        cout << "small not fit" << endl;
    }

    if(hole->isFit(big_peg_adapter)) {
        cout << "big fit" << endl;
    } else {
        cout << "big not fit" << endl;
    }
}