// 对于一个基础类，通过不同的构造方法， 实现不同的内容
// 橘生淮南则为橘，橘生淮北则为枳
#include <iostream>
#include <string>
#include <memory>

using namespace std;

class Car {
public:
    Car(): name_(""), light_(0), elec_(false) {}
    void set_name(string t) {
        name_ = t;
        cout << "set name = " << t << endl;
    }
    void set_light(int num) {
        light_ = num;
        cout << "set num = " << num << endl;
    }
    void set_electron(bool flag) {
        elec_ = flag;
        if(flag) {
            cout << "set flag true" << endl;
        } else {
            cout << "set flag false" << endl;
        }
    }
private:
    string name_;
    int light_;
    bool elec_;
};

class CarBuilder {
public:
    Car getCar() {
        return car_;
    }
    virtual void generateName() = 0;
    virtual void generateLight() = 0;
    virtual void generateElect() = 0;
protected:
    Car car_;
};

class AudiBuilder : public CarBuilder {
public:
    virtual void generateName() override {
        car_.set_name("Audi");
    }
    virtual void generateLight() override {
        car_.set_light(4);
    }
    virtual void generateElect() override {
        car_.set_electron(false);
    }
};

class BenzBuilder : public CarBuilder {
public:
    virtual void generateName() override {
        car_.set_name("Benz");
    }
    virtual void generateLight() override {
        car_.set_light(6);
    }
    virtual void generateElect() override {
        car_.set_electron(true);
    }
};

class Constructor {
public:
    void set_builder(shared_ptr<CarBuilder> b) {
        builder_ = b;
    }

    Car doConstruct() {
        builder_->generateName();
        builder_->generateLight();
        builder_->generateElect();
        return builder_->getCar();
    }
private:
    shared_ptr<CarBuilder> builder_;
};

int main() {
    shared_ptr<CarBuilder> builder;
    unique_ptr<Constructor> constructor(new Constructor());
    Car car;
    // audi
    builder.reset(new AudiBuilder());
    constructor->set_builder(builder);
    car = constructor->doConstruct();

    //benz
    builder.reset(new BenzBuilder());
    constructor->set_builder(builder);
    car = constructor->doConstruct();
}