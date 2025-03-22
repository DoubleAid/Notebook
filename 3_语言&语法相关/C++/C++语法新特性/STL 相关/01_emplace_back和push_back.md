# emplace_back 和 push_back 的异同点

std::vector 的 emplace_back 和 push_back 方法都用于向 std::vector 的末尾添加元素，但它们在实现和性能上有显著的区别。以下是它们的异同点：

## 相同点

功能：

+ 两者都用于向 std::vector 的末尾添加元素。
+ 它们都会在需要时自动扩容。

语法：

+ 两者都接受一个或多个参数，用于构造新元素。

## 不同点

实现方式：

+ push_back：接受一个已经构造好的对象，将其拷贝或移动到 std::vector 的末尾。
+ emplace_back：直接在 std::vector 的末尾构造对象，避免了不必要的拷贝或移动操作。

性能：

+ push_back：可能会涉及额外的拷贝或移动操作，尤其是当传递的对象较大时，性能开销可能较高。
+ emplace_back：直接在目标位置构造对象，避免了不必要的拷贝或移动，性能更高。

参数传递：

+ push_back：接受一个已经构造好的对象，参数类型必须与 std::vector 的元素类型匹配。
+ emplace_back：接受构造函数的参数，直接在目标位置构造对象，参数类型可以是构造函数所需的任何类型。

## 示例代码

### 示例 1：push_back

```cpp
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> vec;

    std::string str = "Hello";
    vec.push_back(str);  // 拷贝 str 到 vector 的末尾

    vec.push_back(std::string("World"));  // 移动临时对象到 vector 的末尾

    for (const auto& s : vec) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    return 0;
}

// 输出
// Hello World
```

### 示例 2：emplace_back

```cpp
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> vec;

    vec.emplace_back("Hello");  // 直接在 vector 的末尾构造对象
    vec.emplace_back(10, 'a');  // 直接在 vector 的末尾构造对象，使用 std::string 的构造函数

    for (const auto& s : vec) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    return 0;
}

// 输出
// Hello aaaaaaaaaa
```

性能对比

假设有一个类 MyClass，其构造函数和拷贝/移动构造函数都有一定的开销：

```cpp
#include <iostream>
#include <vector>

class MyClass {
public:
    MyClass(int x) : value_(x) {
        std::cout << "Constructor called for " << value_ << std::endl;
    }

    MyClass(const MyClass& other) : value_(other.value_) {
        std::cout << "Copy constructor called for " << value_ << std::endl;
    }

    MyClass(MyClass&& other) noexcept : value_(other.value_) {
        std::cout << "Move constructor called for " << value_ << std::endl;
    }

private:
    int value_;
};

int main() {
    std::vector<MyClass> vec;

    std::cout << "Using push_back:" << std::endl;
    vec.push_back(MyClass(10));  // 拷贝或移动构造

    std::cout << "\nUsing emplace_back:" << std::endl;
    vec.emplace_back(20);  // 直接构造

    return 0;
}

// 输出
// Using push_back:
// Constructor called for 10
// Move constructor called for 10

// Using emplace_back:
// Constructor called for 20
```

总结

push_back：

+ 用于将一个已经构造好的对象添加到 std::vector 的末尾。
+ 可能涉及额外的拷贝或移动操作，性能开销可能较高。
+ 参数类型必须与 std::vector 的元素类型匹配。

emplace_back：

+ 直接在 std::vector 的末尾构造对象，避免了不必要的拷贝或移动操作。
+ 性能更高，尤其是在处理大型对象时。
+ 参数类型可以是构造函数所需的任何类型。
在实际使用中，如果需要高效地构造对象并添加到 std::vector 中，推荐使用 emplace_back。

## vec.emplace_back(MyClass(10)); 也就是说其实 MyClass(10) 并没有执行，还是取构造的参数，在向量的末尾用构造函数构造了一个变量

是的，你的理解完全正确！vec.emplace_back(MyClass(10)); 中，MyClass(10) 并没有真正执行构造，而是直接将参数 10 传递给 MyClass 的构造函数，在 std::vector 的末尾直接构造了一个对象。

### 1. emplace_back 的行为

emplace_back 是 std::vector 的一个成员函数，用于直接在容器的末尾构造对象。它接受构造函数的参数，并在目标位置直接构造对象，而不是先构造一个临时对象再移动或拷贝。

语法

```cpp
template <class... Args>
void emplace_back(Args&&... args);
```

Args&&... args：可变参数模板，接受构造函数的参数。

行为：直接在目标位置构造对象，避免了不必要的拷贝或移动操作。

### 2. 示例代码

#### 示例 1：emplace_back

```cpp
#include <iostream>
#include <vector>

class MyClass {
public:
    MyClass(int x) {
        std::cout << "Constructor called for " << x << std::endl;
    }
};

int main() {
    std::vector<MyClass> vec;
    vec.emplace_back(10);  // 直接在 vector 的末尾构造对象
    return 0;
}

// 输出
// Constructor called for 10
```

vec.emplace_back(10); 直接将参数 10 传递给 MyClass 的构造函数，在 std::vector 的末尾构造了一个对象。
MyClass(10) 并没有真正执行构造，而是直接传递参数 10 给构造函数。

### 3. 与 push_back 的对比

#### 示例 2：push_back

```cpp
#include <iostream>
#include <vector>

class MyClass {
public:
    MyClass(int x) {
        std::cout << "Constructor called for " << x << std::endl;
    }
};

int main() {
    std::vector<MyClass> vec;
    vec.push_back(MyClass(10));  // 先构造一个临时对象，然后移动到 vector 的末尾
    return 0;
}

// 输出
// Constructor called for 10
// Constructor called for 10
```

解释

vec.push_back(MyClass(10)); 先构造一个临时对象 MyClass(10)，然后将这个临时对象移动到 std::vector 的末尾。

因此，MyClass(10) 的构造函数被调用了两次：一次用于构造临时对象，一次用于移动构造。

### 4. 性能对比

emplace_back：

直接在目标位置构造对象，仅涉及 1 次构造。
避免了不必要的拷贝或移动操作，性能更高。

push_back：

先构造一个临时对象，然后将这个临时对象移动到 std::vector 的末尾。
涉及 2 次构造（一次构造临时对象，一次移动构造），性能开销可能较高。

### 5. 总结

emplace_back：

直接在目标位置构造对象，避免了不必要的拷贝或移动操作。
参数直接传递给构造函数，仅涉及 1 次构造。
适用于需要高效构造对象的场景。

push_back：

先构造一个临时对象，然后将这个临时对象移动到 std::vector 的末尾。
涉及 2 次构造，性能开销可能较高。
适用于已经构造好的对象。
通过合理使用 emplace_back 和 push_back，可以优化代码的性能，尤其是在处理大型对象时。
