在C语言中， NULL 被定义为空指针 `(void*)0`, C语言中把空指针赋给int和char指针的
时候，发生了隐式类型转换，把void指针转换成了相应类型的指针。
```c++
int  *pi = NULL;
char *pc = NULL;
```
在 C++ 中因为C++是强类型语言，void*是不能隐式转换成其他类型的指针的
```c++
#ifdef __cplusplus
#define NULL 0
#else
#define NULL ((void *)0)
#endif
```
在C++中，NULL实际上是0.因为C++中不能把void*类型的指针隐式转换成其他类型的指针，
所以为了结果空指针的表示问题，C++引入了0来表示空指针，这样就有了上述代码中的NULL宏定义。
C++11加入了nullptr，可以保证在任何情况下都代表空指针，而不会出现上述的情况
```c++
void func(void* i) { cout << "func1" << endl;}
 
void func(int i) { cout << "func2" << endl;}
 
void main(int argc,char* argv[]) {
    func(NULL); // 输出为 func2
    func(nullptr); // 输出为 func1
}
```