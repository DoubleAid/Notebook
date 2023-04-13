// 原型模式（Prototype Pattern）是用于创建重复的对象，同时又能保证性能。 
// 这种类型的设计模式属于创建型模式，它提供了一种创建对象的最佳方式。 
// 这种模式是实现了一个原型接口，该接口用于创建当前对象的克隆。 
// 当直接创建对象的代价比较大时，则采用这种模式。
#include <iostream>
#include <memory>
#include <string>

using namespace std;

class Object {
public:
    virtual Object* clone() = 0;
};

class Attachment {
public:
    void set_content(string s) {
        // 这个过程可能需要大量的计算
        content_ = s;
    }
    string get_content() const {
        return content_;
    }
private:
    string content_;
};

class Email : public Object {
public:
    Email() {}
    Email(string s, string attachment_content) : text_(s) {
        attachment_->set_content(attachment_content);
    }
    ~Email() {
        if(attachment_) delete attachment_;
        attachment_ = nullptr;
    }
    void display() {
        cout << "---------- view Email ---------" << endl;
        cout << "text: " << text_ << endl;
        cout << "attachment: " << attachment_->get_content() << endl;
        cout << "---------- end view ---------" << endl;
    }
    Email* clone() override {
        return new Email(text_, attachment_->get_content());
    }
    void changeText(string s) {
        text_ = s;
    }
    void changeAttachment(string s) {
        attachment_->set_content(s);
    }
private:
    string text_;
    Attachment* attachment_ = nullptr;
};

int main() {
    Email* email = new Email("first text", "first content");
    Email* copy_email = email->clone();
    copy_email->changeText("second text");
    copy_email->changeAttachment("second content");
    cout << "origin mail" << endl;
    email->display();

    cout << "copy mail" << endl;
    copy_email->display();

    delete email;
    delete copy_email;
}