# GDB调试流程

GDB（GNU Debugger）是一个强大的调试工具，常用于调试C、C++等语言的程序。它可以帮助开发者检查程序的运行状态、设置断点、查看变量值、单步执行代码以及分析崩溃原因等。以下是使用GDB进行调试的基本流程：

---

## **1. 编译程序**

在使用GDB调试之前，需要确保程序是用调试信息编译的。通常需要在编译时添加`-g`选项，以便生成调试信息。

```bash
gcc -g -o my_program my_program.c
# 或者对于C++程序
g++ -g -o my_program my_program.cpp
```

### **1.1 在CMakeLists.txt 里添加

```bash
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")  # C++编译选项
set(CMAKE_C_FLAGS_DEBUG "-g -O0")    # C编译选项

# -g：生成调试信息。
# -O0：禁用优化，便于调试。
```

---

## **2. 启动GDB**

使用`gdb`命令启动调试器，并加载要调试的程序。

```bash
gdb my_program
```

如果程序已经运行并崩溃，可以通过`gdb`附加到进程或核心转储文件：

```bash
# 附加到正在运行的进程（通过进程ID）
gdb -p <PID>

# 调试核心转储文件
gdb my_program core
```

---

## **3. 设置断点**

在调试过程中，断点是暂停程序执行的关键位置。可以使用`break`命令设置断点。

```gdb
# 设置断点
break main          # 在main函数入口处设置断点
break file.c:10     # 在file.c文件的第10行设置断点
break func_name     # 在函数func_name入口处设置断点
```

---

## **4. 运行程序**

使用`run`命令启动程序。如果程序需要命令行参数，可以在`run`后面添加参数。

```gdb
run arg1 arg2
```

---

## **5. 暂停和恢复程序**

- **暂停程序**：程序会在遇到断点时自动暂停。也可以使用`Ctrl+C`手动暂停正在运行的程序。
- **恢复程序**：使用`continue`命令从断点处继续运行程序。

```gdb
continue
```

---

## **6. 单步执行**

- **单步执行（逐语句）**：使用`next`命令逐语句执行代码。当遇到函数调用时，会直接跳过。
  
  ```gdb
  next
  ```

- **单步进入（逐指令）**：使用`step`命令逐指令执行代码。当遇到函数调用时，会进入函数内部。
  
  ```gdb
  step
  ```

- **退出函数**：使用`finish`命令执行完当前函数并返回到调用处。
  
  ```gdb
  finish
  ```

---

## **7. 查看变量和内存**

- **查看变量值**：
  
  ```gdb
  print variable_name
  ```

- **查看内存内容**：
  
  ```gdb
  x/格式 地址
  x/10gx 0x12345678  # 查看从地址0x12345678开始的10个8字节内容
  ```

- **查看寄存器值**：
  
  ```gdb
  info registers
  ```

---

## **8. 查看调用栈**

当程序暂停时，可以查看调用栈信息，以了解程序的执行路径。

```gdb
backtrace  # 或 bt
```

---

## **9. 修改变量值**

在调试过程中，可以修改变量的值以测试不同的情况。

```gdb
set variable_name = new_value
```

---

## **10. 分析崩溃原因**

如果程序崩溃，GDB会自动暂停执行。可以使用以下命令分析崩溃原因：

- **查看崩溃点**：
  
  ```gdb
  backtrace
  ```

- **查看崩溃时的变量和内存**：
  
  ```gdb
  print variable_name
  x/10gx $rsp  # 查看栈内存
  ```

---

## **11. 删除断点**

如果不再需要某个断点，可以使用`delete`命令删除它。

```gdb
delete 1  # 删除编号为1的断点
delete    # 删除所有断点
```

---

## **12. 退出GDB**

完成调试后，可以使用`quit`命令退出GDB。

```gdb
quit
```

---

## **总结**

GDB调试流程主要包括编译程序、启动GDB、设置断点、运行程序、单步执行、查看变量和内存、分析崩溃原因以及退出调试器。
通过这些步骤，可以有效地定位和修复程序中的问题。
