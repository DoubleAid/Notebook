# C++强制类型转换操作符 static_cast
+ static_cast是一个强制类型转换操作符。强制类型转换，也称为显式转换，C++中强制类型转换操作符有
`static_cast`、`dynamic_cast`、`const_cast`、`reinterpert_cast`四个。本节介绍static_cast操作符。
+ 编译器隐式执行的任何类型转换都可以由static_cast来完成，比如int与float、double与char、enum与int之间的转换等。

```c++
double a = 1.999;
int b = static_cast<double>(a); //相当于a = b ;
```
当编译器隐式执行类型转换时，大多数的编译器都会给出一个警告：
```
e:\vs 2010 projects\static_cast\static_cast\static_cast.cpp(11): warning C4244: “初始化”: 从“double”转换到“int”，可能丢失数据
　　使用static_cast可以明确告诉编译器，这种损失精度的转换是在知情的情况下进行的，也可以让阅读程序的其他程序员明确你转换的目的而不是由于疏忽。
```
把精度大的类型转换为精度小的类型，static_cast使用位截断进行处理。

使用static_cast可以找回存放在void*指针中的值。
```c++
double a = 1.999;
void * vptr = & a;
double * dptr = static_cast<double*>(vptr);
cout<<*dptr<<endl;//输出1.999
```
static_cast也可以用在于基类与派生类指针或引用类型之间的转换。然而它不做运行时的检查，不如dynamic_cast安全。static_cast仅仅是依靠类型转
换语句中提供的信息来进行转换，而dynamic_cast则会遍历整个类继承体系进行类型检查,因此dynamic_cast在执行效率上比static_cast要差一些。现在我们有父类与其派生类如下：

```c++
class ANIMAL {
public:
    ANIMAL():_type("ANIMAL"){};
    virtual void OutPutname(){cout<<"ANIMAL";};
private:
    string _type ;
};

class DOG:public ANIMAL {
public:
    DOG():_name("大黄"),_type("DOG"){};
    void OutPutname(){cout<<_name;};
    void OutPuttype(){cout<<_type;};
private:
    string _name ;
    string _type ;
};
```
此时我们进行派生类与基类类型指针的转换：注意从下向上的转换是安全的，从上向下的转换不一定安全。
```c++
int main()
{
    //基类指针转为派生类指针,且该基类指针指向基类对象。
    ANIMAL * ani1 = new ANIMAL ;
    DOG * dog1 = static_cast<DOG*>(ani1);
    //dog1->OutPuttype();//错误，在ANIMAL类型指针不能调用方法OutPutType（）；在运行时出现错误。

    //基类指针转为派生类指针，且该基类指针指向派生类对象
    ANIMAL * ani3 = new DOG;
    DOG* dog3 = static_cast<DOG*>(ani3);
    dog3->OutPutname(); //正确

    //子类指针转为派生类指针
    DOG *dog2= new DOG;
    ANIMAL *ani2 = static_cast<DOG*>(dog2);
    ani2->OutPutname(); //正确，结果输出为大黄

    //
    system("pause");

}
```
static_cast可以把任何类型的表达式转换成void类型。
static_cast把任何类型的表达式转换成void类型。
另外，与const_cast相比，static_cast不能把换掉变量的const属性，也包括 volitale 或者__unaligned属性。