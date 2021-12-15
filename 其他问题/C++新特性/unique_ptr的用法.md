smart pointer 是 C++ 11 用来替代 指针 的新特性
C++11的 SmartPointer有三个：
+ std::unique_ptr
+ std::shared_ptr
+ std::weak_ptr

```cpp
// 不使用 unique_ptr
bool WorthBuying()
{
    TeaShopOwner* the_owner = new TeaShopOwner();
    
    if (the_owner->SupportCCP()) {
        delete the_owner;
        return false;
    }
    
    if (!the_owner->AdmitTaiwanAsACountry()) {
        delete the_owner;
        return false;
    }
    
    delete the_owner;
    return true;
}

// 使用 unique_ptr
bool WorthBuying()
{
    std::unique_ptr<TeaShopOwner> the_owner = std::make_unique<TeaShopOwner>();
    if (the_owner->SupportCCP())
        return false;
    if (!the_owner->AdmitTaiwanAsACountry())
        return false;
    return true;
}
```
**注意**：std::make_unique仅支持C++14


```cpp
// 不使用 unique_ptr
TeaShopOwnder* CreateOwner();

{
    TeaShopOwner* the_owner = CreateOwner();
    // Do something with the_owner
    delete the_owner;
}

// 使用 unique_ptr
std::unique_ptr<TeaShopOwner> CreateOwner();

{
    std::unique_ptr<TeaShopOwner> the_owner = CreateOwner();
    // Do something with the_owner
}
```

shared_ptr使用引用计数，每一个shared_ptr的拷贝都指向相同的内存。每使用他一次，内部的引用计数加1，每析构一次，内部的引用计数减1，减为0时，删除所指向的堆内存。shared_ptr内部的引用计数是安全的，但是对象的读取需要加锁。

```cpp
#include "stdafx.h"
#include <iostream>
#include <future>
#include <thread>

using namespace std;
class Person
{
public:
    Person(int v) {
        value = v;
        std::cout << "Cons" <<value<< std::endl;
    }
    ~Person() {
        std::cout << "Des" <<value<< std::endl;
    }

    int value;

};

int main()
{
    std::shared_ptr<Person> p1(new Person(1));// Person(1)的引用计数为1

    std::shared_ptr<Person> p2 = std::make_shared<Person>(2);

    p1.reset(new Person(3));// 首先生成新对象，然后引用计数减1，引用计数为0，故析构Person(1)
                            // 最后将新对象的指针交给智能指针

    std::shared_ptr<Person> p3 = p1;//现在p1和p3同时指向Person(3)，Person(3)的引用计数为2

    p1.reset();//Person(3)的引用计数为1
    p3.reset();//Person(3)的引用计数为0，析构Person(3)
    return 0;
}
```