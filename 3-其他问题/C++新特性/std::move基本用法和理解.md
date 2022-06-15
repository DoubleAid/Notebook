+ **场景**: 
    + C++ 标准库使用比如vector::push_back 等这类函数时,会对参数的对象进行复制,连数据也会复制.这就会造成对象内存的额外创建, 本来原意是想把参数push_back进去就行了.
    + C++11 提供了std::move 函数来把左值转换为xrvalue, 而且新版的push_back也支持&&参数的重载版本,这时候就可以高效率的使用内存了.
    + 对指针类型的标准库对象并不需要这么做.

+ 参考:
  1. Move Constructors and Move Assignment Operators (C++)
  2. std::move

+ 说明:

    std::move(t) 用来表明对象t 是可以moved from的,它允许高效的从t资源转换到lvalue上.
    注意,标准库对象支持moved from的左值在moved 之后它的对象原值是有效的(可以正常析构),但是是unspecified的,可以理解为空数据,但是这个对象的其他方法返回值不一定是0,比如size().所以,moved from 之后的对象最好还是不要使用吧?(如有不正确理解,请告知)
    对本身进行move,并赋值给本身是undefined的行为.

    ```cpp
    std::vector<int> v = {2, 3, 4}
    v = std::move(v); // undefined behavior
    ```

+ 用法例子
  + 例 1：
    ```cpp
    void TestSTLObject() {
        std::string str = "hello";
        std::vector<std::string> v;

        v.push_back(str);
        std::cout << "After copy, str is \"" << str <<"\"\n";

        v.push_back(std::move(str));
        std::cout << "After move, str is \"" << str << "\"\n";

        std::cout << "the contents of the vector are \"" << v[0] << "\", \"" << v[1] << "\"\n";
    }
    ```
    输出
    ```
    After copy, str is "Hello"
    After move, str is ""
    The contents of the vector are "Hello", "Hello"
    ```
