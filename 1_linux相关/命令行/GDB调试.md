## 编译
```shell
g++ test.cpp -o test.o -g
```

## 启动GDB
```shell
gdb test.o -q # -q 标识 quite
```

## 调试指定
+ break xxx 或 b xxx
在源代码指定的某一行设置断点，其中 xxx 用于指定具体打断点的位置。
+ run 或 r 
执行被调试的程序，其会自动在第一个断点处暂停执行。
+ continue 或 c
当程序在某一断点处停止运行后，使用该指令可以继续执行，直至遇到下一个断点或者程序结束。
+ next 或 n
令程序一行代码一行代码的执行。
+ print xxx 或 p xxx
  打印指定变量的值，其中 xxx 指的就是某一变量名。
+ list 或 l
  显示源程序代码的内容，包括各行代码所在的行号。
+ quit 或 q
  终止调试。