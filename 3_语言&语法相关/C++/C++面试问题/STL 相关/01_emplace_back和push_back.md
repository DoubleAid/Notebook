## 异同点
+ 如果参数是左值，两个调用的都是copy constructor
+ 如果参数是右值，两个调用的都是move constructor（C++ 11后push_back也支持右值）
+ 最主要的区别是，emplace_back支持in-place construction，也就是说emplace_back(10, “test”)
可以只调用一次constructor， 而push_back(MyClass(10, “test”))必须多一次构造和析构

两者都支持左值和右值，emplace_back的优势是右值时的效率优化。这是最大的误解，
emplace_back的最大优势是它可以直接在vector的内存中去构建对象，不用在外面构造完
了再copy或者move进去！！！