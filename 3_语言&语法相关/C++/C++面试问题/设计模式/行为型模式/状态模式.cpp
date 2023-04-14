// 在状态模式（State Pattern）中，类的行为是基于它的状态改变的。
// 意图：允许对象在内部状态发生改变时改变它的行为，对象看起来好像修改了它的类。
// 主要解决：对象的行为依赖于它的状态（属性），并且可以根据它的状态改变而改变它的相关行为。
// 何时使用：代码中包含大量与对象状态有关的条件语句。
// 如何解决：将各种具体的状态类抽象出来。
// 关键代码：通常命令模式的接口中只有一个方法。而状态模式的接口中有一个或者多个方法。而且，状态模式的实现类的方法，
//          一般返回值，或者是改变实例变量的值。也就是说，状态模式一般和对象的状态有关。实现类的方法有不同的功能，
//          覆盖接口中的方法。状态模式和命令模式一样，也可以用于消除 if...else 等条件选择语句。
// 应用实例： 1、打篮球的时候运动员可以有正常状态、不正常状态和超常状态。 
//          2、曾侯乙编钟中，'钟是抽象接口','钟A'等是具体状态，'曾侯乙编钟'是具体环境（Context）。
// 优点： 1、封装了转换规则。 
//       2、枚举可能的状态，在枚举状态之前需要确定状态种类。 
//       3、将所有与某个状态有关的行为放到一个类中，并且可以方便地增加新的状态，只需要改变对象状态即可改变对象的行为。 
//       4、允许状态转换逻辑与状态对象合成一体，而不是某一个巨大的条件语句块。 
//       5、可以让多个环境对象共享一个状态对象，从而减少系统中对象的个数。
// 缺点： 1、状态模式的使用必然会增加系统类和对象的个数。 
//  2、状态模式的结构与实现都较为复杂，如果使用不当将导致程序结构和代码的混乱。 
//  3、状态模式对"开闭原则"的支持并不太好，对于可以切换状态的状态模式，增加新的状态类需要修改那些负责状态转换的源代码，
//  否则无法切换到新增状态，而且修改某个状态类的行为也需修改对应类的源代码。
// 使用场景： 1、行为随状态改变而改变的场景。 2、条件、分支语句的代替者。
// 注意事项：在行为受状态约束的时候使用状态模式，而且状态不超过 5 个。
#include <iostream>
#include <string>

using namespace std;

class AbstractState;
// 论坛账号
class ForumAccount {
 public:
    explicit ForumAccount(std::string name);
    void set_state(std::shared_ptr<AbstractState> state) {
        state_ = state;
    }
    std::shared_ptr<AbstractState> get_state() {
        return state_;
    }
    std::string get_name() {
        return name_;
    }
    void downloadFile(int score);
    void writeNote(int score);
    void replyNote(int score);
 private:
    std::shared_ptr<AbstractState> state_;
    std::string name_;
};

ForumAccount::ForumAccount(std::string name)
    : name_(name), state_(std::make_shared<PrimaryState>(this)) {
    printf("账号%s注册成功!\n", name.c_str());
}
​
void ForumAccount::downloadFile(int score) {
    state_->downloadFile(score);
}
​
void ForumAccount::writeNote(int score) {
    state_->writeNote(score);
}
​
void ForumAccount::replyNote(int score) {
    state_->replyNote(score);
}

class AbstractState {
 public:
    virtual void checkState() = 0;
​
    void set_point(int point) {
        point_ = point;
    }
    int get_point() {
        return point_;
    }
    void set_state_name(std::string name) {
        state_name_ = name;
    }
    std::string get_state_name() {
        return state_name_;
    }
    ForumAccount* get_account() {
        return account_;
    }
​
    virtual void downloadFile(int score) {
        printf("%s下载文件, 扣除%d积分。\n", account_->get_name().c_str(), score);
        point_ -= score;
        checkState();
        printf("%s剩余积分为%d, 当前级别为%s。\n", account_->get_name().c_str(), point_, account_->get_state()->get_state_name().c_str());
    }
​
    virtual void writeNote(int score) {
        printf("%s发布留言, 增加%d积分。\n", account_->get_name().c_str(), score);
        point_ += score;
        checkState();
        printf("%s剩余积分为%d, 当前级别为%s。\n", account_->get_name().c_str(), point_, account_->get_state()->get_state_name().c_str());
    }
​
    virtual void replyNote(int score) {
        printf("%s回复留言, 增加%d积分。\n", account_->get_name().c_str(), score);
        point_ += score;
        checkState();
        printf("%s剩余积分为%d, 当前级别为%s。\n", account_->get_name().c_str(), point_, account_->get_state()->get_state_name().c_str());
    }
​
 protected:
    ForumAccount* account_;
    int point_;
    std::string state_name_;
};