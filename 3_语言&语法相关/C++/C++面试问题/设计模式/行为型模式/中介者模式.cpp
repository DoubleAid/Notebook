// 中介者模式（Mediator Pattern）是用来降低多个对象和类之间的通信复杂性。这种模式提供了一个中介类，
// 该类通常处理不同类之间的通信，并支持松耦合，使代码易于维护。中介者模式属于行为型模式。
// 意图：用一个中介对象来封装一系列的对象交互，中介者使各对象不需要显式地相互引用，从而使其耦合松散，
// 而且可以独立地改变它们之间的交互。
// 主要解决：对象与对象之间存在大量的关联关系，这样势必会导致系统的结构变得很复杂，同时若一个对象发生改变，
// 我们也需要跟踪与之相关联的对象，同时做出相应的处理。
// 何时使用：多个类相互耦合，形成了网状结构。
// 如何解决：将上述网状结构分离为星型结构。
// 关键代码：对象 Colleague 之间的通信封装到一个类中单独处理。
// 应用实例： 1、中国加入 WTO 之前是各个国家相互贸易，结构复杂，现在是各个国家通过 WTO 来互相贸易。 
//          2、机场调度系统。 
//          3、MVC 框架，其中C（控制器）就是 M（模型）和 V（视图）的中介者。
// 优点： 1、降低了类的复杂度，将一对多转化成了一对一。 2、各个类之间的解耦。 3、符合迪米特原则
// 缺点：中介者会庞大，变得复杂难以维护。
// 使用场景： 1、系统中对象之间存在比较复杂的引用关系，导致它们之间的依赖关系结构混乱而且难以复用该对象。 
//          2、想通过一个中间类来封装多个类中的行为，而又不想生成太多的子类。
// 注意事项：不应当在职责混乱的时候使用。

// 房东，租客 和 中介
#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Person;

class Mediator {
public:
    virtual void registerMethod(Person*) = 0;
    virtual void operation(Person*) = 0;
};

class Agency : public Mediator {
public:
    void registerMethod(Person* person) override {
        switch (person->get_person_type()) {
            case kLandlord:
                landlord_list_.push_back(reinterpret_cast<Landlord*>(person));
                break;
            case kTenant:
                tenant_list_.push_back(reinterpret_cast<Tenant*>(person));
                break;
            default:
                break;
        }
    }
    void operation(Person* person) {
        switch (person->get_person_type()) {
            case kLandlord:
                for(int i = 0; i < tenant_list_.size(); i++) {
                    tenant_list_[i]->answer();
                }
                break;
            case kTenant:
                for(int i = 0; i < landlord_list_.size(); i++) {
                    landlord_list_[i]->answer();
                }
                break;
            default:
                break;
        }
    }
private:
    vector<Landlord*> landlord_list_;
    vector<Tenant*> tenant_list_;
};

enum Person_Type {
    kUnknown,
    kLandlord,
    kTenant,
};

class Person {
public:
    void set_mediator(Mediator* m) { mediator_ = m; }
    Mediator* get_mediator() { return mediator_; }
    void set_person_type(Person_Type ty) { person_type_ = ty; }
    Person_Type get_person_type() { return person_type_; }
    virtual void ask() = 0;
    virtual void answer() = 0;
private:
    Mediator* mediator_;
    Person_Type person_type_;
};

class Landlord : public Person {
public:
    Landlord() {
        name_ = "unknown";
        price_ = -1;
        address_ = "unknown";
        phone_number_ = "unknown";
        set_person_type(kUnknown);
    }
    Landlord(string name, int price, string address, string phone_number) : 
            name_(name), price_(price), address_(address), phone_number_(phone_number) {
                set_person_type(kLandlord);
            }
    void answer() override {
        cout << "房东姓名:" << name_ << " 房租：" << price_ 
                << " 地址:" << address_ << " 电话:" << phone_number_ << endl; 
    }
    void ask() override {
        cout << "房东"<< name_ << "查看租客信息：" << endl;
        this->get_mediator()->operation(this);
    }
private:
    string name_;
    int price_;
    string address_;
    string phone_number_;
};

class Tenant : public Person {
public:
    Tenant() : name_("unknown") {}
    explicit Tenant(string name) {
        name_ = name;
        set_person_type(kTenant);
    }
    void ask() override {
        cout << "租客" << name_ << "查看房东信息：" << endl;
        this->get_mediator()->operation(this);
    }
    void answer() override {
        cout << "租客姓名为" << name_ << endl;
    }
private:
    string name_;
};

int main() {
    Agency *mediator = new Agency();

    Landlord* l1 = new Landlord("张三", 1820, "天津", "133333");
    Landlord* l2 = new Landlord("李四", 2020, "苏州", "125333");
    Landlord* l3 = new Landlord("王五", 1920, "上海", "133999");

    l1->set_mediator(mediator);
    l2->set_mediator(mediator);
    l3->set_mediator(mediator);

    mediator->registerMethod(l1);
    mediator->registerMethod(l2);
    mediator->registerMethod(l3);

    Tenant* t1 = new Tenant("zhang");
    Tenant* t2 = new Tenant("zhao");
    t1->set_mediator(mediator);
    t2->set_mediator(mediator);
    mediator->registerMethod(t1);
    mediator->registerMethod(t2);

    t1->ask();
    l1->ask();

    delete mediator;
    delete l1;
    delete l2;
    delete l3;
    delete t1;
    delete t2;
}