对于 main 函数，如果不需要读取任何命令时，是不需要参数的，但如果需要参数读取时，需要加上 argc 和 argv 两个参数
以一下代码为例
```cpp
#include <iostream>
using namespace std;
int main(int argc, char** argv) {
    cout << tostring(argc) << endl;
    for (int i=0; i<argc; i++) {
        cout << argv[i] << endl;
    }
    return 0;
}
```

argc 是 argument count 的缩写，表示传入 main 函数的参数的个数
argv 是 argument vector 的缩写， 表示传入 main 函数的参数序列或指针， 并且 第一个参数 argv[0] 一定是程序的名称， 并且包含了程序所在的完整路径。

C/C++ 中的main 函数， 经常带有参数 argc 和 argv 有两种形式
+ int main(int argc, char **argv)
+ int main(int argc, char *argv[])

例如以下代码
```cpp
#include <stdio.h>

int main(int argc, char *argv[]) {
    for (int i = 0; i < argc; i++) {
        printf("Argument %d is %s \n", i, argv[i]);
    }
    return 0;
}
```
以如下命令编译运行
```
gcc hello.c
./a.out a b c
```
运行结果如下
```cpp
Argument 0 is ./a.out
Argument 1 is a
Argument 2 is b
Argument 3 is c
```
