### CMakeLists.txt的基本结构
编写`CMakeList.txt`最常见的功能就是调用其他的 .h 头文件和 .so/.a 库文件， 将 .cpp/.c/.cc 文件编译成可执行文件的库文件。

最常见的命令如下：
```python
# 本 CMakeLists.txt的project名称
# 会自动创建两个变量， PROJECT_SOURCE_DIR 和 PROJECT_NAME
# ${PROJECT_SOURCE_DIR}: 本CMakeLists.txt所在的文件夹路径
# ${PROJECT_NAME}: 本CMakeList.txt的project名称
project(xxx)

# 获取路径下的所有 .cpp/.c/.cc 文件，并赋值给变量
aux_source_directory(路径 变量)

# 给文件名/路径或者其他字符串其别名，用 ${变量}获取变量的内容
set(变量 文件名/路径/...)

# 添加编译选项
add_definitions(编译选项)

# 打印消息
message(消息)

# 编译子文件夹的CMakeLists.txt
add_subdirectory(子文件夹名称)

# 将.cpp/.c/.cc文件生成可执行文件
add_executable(可执行文件名称 文件)

# 规定.so .a库文件的路径
link_directories(路径)

# 对add_library或add_executable生成的文件进行链接操作
# 注意，通常库文件的名称为libxxx.so， 在这里只要写xxx即可
target_link_libraries(库文件名称/可执行文件名称 链接的库文件名称)

```

### 实际工程例子
#### <font color=chocolate>工程结构</font>
```
| --- example_person.cpp
| --- proto_pb2
|       | --- Person.pb.cc
|       | --- Person.pb.h
|
| --- proto_buf
|       | --- General_buf_read.h
|       | --- General_buf_wirte.h
|
| --- protobuf
        | --- bin
        |       | --- ...
        |
        | --- include
        |       | --- ...
        |
        | --- lib
                | --- ...
```

**目录结构含义：**
+ **protobuf**：Google提供的相关解析库和头文件，被proto_pb2文件夹内引用；
+ **proto_pb2**：封装的Person结构和Person相关的处理函数，被proto_buf文件夹内引用；
+ **proto_buf**：封装的read和write函数，被example_persom.cpp文件引用。

也就是说：
example_person.cpp -> proto_buf文件夹 -> proto_pb2 -> protobuf文件夹

#### <font color=chocolate>创建CMakeLists.txt</font>
本项目的CMakeLists.txt的文件数量是2个，目录层次结构为上下层关系。通常的解决方案，就是将下层目录编译成一个静态库文件，让上层目录直接读取和调用，而上层目录就直接生教程上层目录就是生成一个可执行文件。

上层CMakeLists.txt 的内容为
```python
cmake_minimum_required(VERSION 3.0)
project(example_person)

# 如果代码需要支持 C++11， 就直接加上这句
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++0x")

# 如果想要生成的可执行文件拥有符号表，可以gdb调试， 就直接加上这句
add_definitions("-Wall -g")

# 设置变量， 下面的代码都可以用到
set(GOOGLE_PROTOBUF_DIR ${PROJECT_SOURCE_DIR}/protobuf)
set(PROTO_PB_DIR ${PROJECT_SOURCE_DIR}/proto_pb2)
set(PROTO_BUF_DIR $PROJECT_SOURCE_DIR}/proto_buf)

# 编译子文件夹的 CMakeLists.txt
add_subdirectory(proto_pb2)

# 规定头文件的路径
include_directory(${PROJECT_SOURCE_DIR}
    &{PROTO_PB_DIR} ${PROTO_BUF_DIR}
)

# 生成可执行文件
add_executable(${PROJECT_NAME} example_person.cpp)

# 链接操作
target_link_libraries(${PROJECT_NAME} general_pb2)

```
这一段可能看不到两个地方，第一是链接操作的general_pb2，第二是按照上文的CMakeLists.txt的流程，并没有规定link_directories的库文件地址啊，这是什么道理？

这两个其实是一个道理，add_subdirectory起到的作用！

当运行到add_subdirectory这一句时，会先将子文件夹进行编译，而libgeneral_pb2.a是在子文件夹中生成出来的库文件。子文件夹运行完后，父文件夹就已经知道了libgeneral_pb2.a这个库，因而不需要link_directories了。

同时，另一方面，在add_subdirector之前set的各个变量，在子文件夹中是可以调用的！

下层CMakeLists.txt的内容为：

```java
project(general_pb2)

aux_source_directory(${PROJECT_SOURCE_DIR} PB_FILES)

add_library(${PROJECT_NAME} STATIC ${PB_FILES})

include_directories(${PROJECT_SOURCE_DIR} ${GOOGLE_PROTOBUF_DIR}/include)

link_directoies(${GOOGLE_PROTOBUF_DIR/lib/)

target_link_libraries(${PROJECT_NAME}
    protobuf
)
```
在这里，GOOGLE_PROTOBUF_DIR是上层CMakeLists.txt中定义的，libprotobuf.a是在${GOOGLE_PROTOBUF_DIR}/lib/目录下的。

显然可见，这就是一个标准的CMakeLixts.txt的流程。

#### <font color=chocolate>CMakeLists.txt的编译</font>

```
$ mkdir build && cd build
$ cmake
$ make
```

最终生成可执行文件example_person。

可以通过以下命令来运行该可执行文件：
```
./example_person
```

### 参考链接
+ https://blog.csdn.net/qq_38410730/article/details/102477162
+ 