// 在访问者模式（Visitor Pattern）中，我们使用了一个访问者类，它改变了元素类的执行算法。通过这种方式，
// 元素的执行算法可以随着访问者改变而改变。这种类型的设计模式属于行为型模式。根据模式，元素对象已接受访问者对象，
// 这样访问者对象就可以处理元素对象上的操作。
// 意图：主要将数据结构与数据操作分离。
// 主要解决：稳定的数据结构和易变的操作耦合问题。
// 何时使用：需要对一个对象结构中的对象进行很多不同的并且不相关的操作，而需要避免让这些操作"污染"这些对象的类，
//          使用访问者模式将这些封装到类中。
// 如何解决：在被访问的类里面加一个对外提供接待访问者的接口。
// 关键代码：在数据基础类里面有一个方法接受访问者，将自身引用传入访问者。
// 应用实例：您在朋友家做客，您是访问者，朋友接受您的访问，您通过朋友的描述，然后对朋友的描述做出一个判断，这就是访问者模式。
// 优点： 1、符合单一职责原则。 2、优秀的扩展性。 3、灵活性。
// 缺点： 1、具体元素对访问者公布细节，违反了迪米特原则。 2、具体元素变更比较困难。 
//       3、违反了依赖倒置原则，依赖了具体类，没有依赖抽象。
// 使用场景： 1、对象结构中对象对应的类很少改变，但经常需要在此对象结构上定义新的操作。 
//          2、需要对一个对象结构中的对象进行很多不同的并且不相关的操作，
//          而需要避免让这些操作"污染"这些对象的类，也不希望在增加新操作时修改这些类。
// 注意事项：访问者可以对功能进行统一，可以做报表、UI、拦截器与过滤器。
#include <iostream>
#include <string>

using namespace std;

class Apple;
class Book;
// 抽象访问者
class Vistor {
public:
void set_name(std::string name) {
    name_ = name;
}
​virtual void visit(Apple *apple) = 0;
virtual void visit(Book *book) = 0;
protected:
    std::string name_;
};

// 具体访问者类: 顾客
class Customer : public Vistor {
 public:
    void visit(Apple *apple) {
        std::cout << "顾客" << name_ << "挑选苹果。" << std::endl;
    }
​
    void visit(Book *book) {
        std::cout << "顾客" << name_ << "买书。" << std::endl;
    }
};

// 具体访问者类： 收银员
class Saler : public Vistor {
 public:
    void visit(Apple *apple) {
        std::cout << "收银员" << name_ << "给苹果过称, 然后计算价格。" << std::endl;
    }
​
    void visit(Book *book) {
        std::cout << "收银员" << name_ << "计算书的价格。" << std::endl;
    }
};
​
class Product {
 public:
    virtual void accept(Vistor *vistor) = 0;
};

class Apple : public Product {
 public:
    void accept(Vistor *vistor) override {
        vistor->visit(this);
    }
};

class Book : public Product {
 public:
    void accept(Vistor *vistor) override {
        vistor->visit(this);
    }
};

class ShoppingCart {
 public:
    void accept(Vistor *vistor) {
        for (auto prd : prd_list_) {
            prd->accept(vistor);
        }
    }
​
    void addProduct(Product *product) {
        prd_list_.push_back(product);
    }
​
    void removeProduct(Product *product) {
        prd_list_.remove(product);
    }
private:
    std::list<Product*> prd_list_;
};

int main() {
    Book book;
    Apple apple;
    ShoppingCart basket;
​
    basket.addProduct(&book);
    basket.addProduct(&apple);
​
    Customer customer;
    customer.set_name("小张");
    basket.accept(&customer);
​
    Saler saler;
    saler.set_name("小杨");
    basket.accept(&saler);
​
    return 0;
}