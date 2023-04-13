// 顾名思义，责任链模式（Chain of Responsibility Pattern）为请求创建了一个接收者对象的链。
// 这种模式给予请求的类型，对请求的发送者和接收者进行解耦。这种类型的设计模式属于行为型模式。
// 在这种模式中，通常每个接收者都包含对另一个接收者的引用。如果一个对象不能处理该请求，
// 那么它会把相同的请求传给下一个接收者，依此类推。

// 下面的例子 员工申请处理票据需要上报上级， 如果上级处理不了， 就上报给上级的上级

#include <iostream>

using namespace std;

class ApproverInterface {
public:
    virtual void setSuperior(ApproverInterface* superior) = 0;
    virtual void handleRequest(double amount) = 0;
};

class BaseApprover : public ApproverInterface {
public:
    BaseApprover(double mpa, string n): 
            max_processible_amount_(mpa), name_(n), superior_(nullptr) {}
    void setSuperior(ApproverInterface* superior) {
        superior_ = superior;
    }
    void handleRequest(double amount) {
        if(amount < max_processible_amount_) {
            cout << name_ << "处理了该票据， 票面额度为" << amount << endl;
            return;
        }
        if(superior_ != nullptr) {
            cout << name_ << "无法处理， 已经移交给上级处理" << endl;
            superior_->handleRequest(amount);
            return;
        }
        cout << "无人处理该票据， 票据金额为" << amount << endl;
    }
private:
    double max_processible_amount_;
    string name_;
    ApproverInterface* superior_;
};

class GroupLeader: public BaseApprover {
public:
    explicit GroupLeader(string name) : BaseApprover(10, name) {}
};

class Manager: public BaseApprover {
public:
    explicit Manager(string name) : BaseApprover(100, name) {}
};

class Boss : public BaseApprover {
public:
    explicit Boss(string name) : BaseApprover(1000, name) {}
};

int main() {
    GroupLeader* group_leader = new GroupLeader("张组长");
    Manager* manager = new Manager("李经理");
    Boss* boss = new Boss("王老板");

    group_leader->setSuperior(manager);
    manager->setSuperior(boss);

    group_leader->handleRequest(8);
    group_leader->handleRequest(88);
    group_leader->handleRequest(888);
    group_leader->handleRequest(8888);

    delete group_leader;
    delete manager;
    delete boss;
}