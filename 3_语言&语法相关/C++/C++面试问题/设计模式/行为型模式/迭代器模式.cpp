// 迭代器模式（Iterator Pattern）是 Java 和 .Net 编程环境中非常常用的设计模式。
// 这种模式用于顺序访问集合对象的元素，不需要知道集合对象的底层表示。
// 意图：提供一种方法顺序访问一个聚合对象中各个元素, 而又无须暴露该对象的内部表示。
// 主要解决：不同的方式来遍历整个整合对象。
// 何时使用：遍历一个聚合对象。
// 如何解决：把在元素之间游走的责任交给迭代器，而不是聚合对象。
// 关键代码：定义接口：hasNext, next。
// 应用实例：JAVA 中的 iterator。
// 优点： 1、它支持以不同的方式遍历一个聚合对象。 
//       2、迭代器简化了聚合类。 
//       3、在同一个聚合上可以有多个遍历。 
//       4、在迭代器模式中，增加新的聚合类和迭代器类都很方便，无须修改原有代码。
// 缺点：由于迭代器模式将存储数据和遍历数据的职责分离，增加新的聚合类需要对应增加新的迭代器类，
//      类的个数成对增加，这在一定程度上增加了系统的复杂性。
// 使用场景： 1、访问一个聚合对象的内容而无须暴露它的内部表示。 
//          2、需要为聚合对象提供多种遍历方式。 
//          3、为遍历不同的聚合结构提供一个统一的接口。
// 注意事项：迭代器模式就是分离了集合对象的遍历行为，抽象出一个迭代器类来负责，
//          这样既可以做到不暴露集合的内部结构，又可让外部代码透明地访问集合内部的数据。

#include <iostream>
#include <string>
#include <vector>

using namespace std;

// 抽象迭代器
class TVIterator {
public:
    virtual void setChannel(int i) = 0;
    virtual void next() = 0;
    virtual void previous() = 0;
    virtual bool isLast() = 0;
    virtual string currentChannel() = 0;
    virtual bool isFirst() = 0;
};

// 具象迭代器
class SKIterator : public TVIterator {
public:
    explicit SKIterator(vector<string> tvs) : tvs_(tvs) {}
    void next() override {
        if(current_index_ < tvs_.size()) current_index_++;
    }
    void previous() override {
        if(current_index_ > 0) current_index_--;
    }
    void setChannel(int i) override {
        current_index_ = i;
    }
    string currentChannel() override {
        return tvs_[current_index_];
    }
    bool isLast() override {
        return current_index_ == tvs_.size();
    }
    bool isFirst() override {
        return current_index_ == 0;
    }
private:
    vector<string> tvs_;
    int current_index_ = 0;
};

// 抽象集合
class Television {
public:
    virtual shared_ptr<TVIterator> createIterator() = 0;
};

// 具体集合
class SKTelevision : public Television {
public:
    shared_ptr<TVIterator> createIterator() {
        return make_shared<SKIterator>(tvs_);
    }
    void addItem(string item) {
        tvs_.push_back(item);
    }
private:
    vector<string> tvs_;
};

int main() {
    SKTelevision stv;
    stv.addItem("CCTV-1");
    stv.addItem("CCTV-2");
    stv.addItem("CCTV-3");
    stv.addItem("CCTV-4");
    stv.addItem("CCTV-5");

    auto iter = stv.createIterator();
    while (!iter->isLast()) {
        cout << iter->currentChannel() << endl;
        iter->next();
    }
    return 0;
}