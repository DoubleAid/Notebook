[参考链接](https://zhuanlan.zhihu.com/p/97369704)
# 使用 find_package 引入外部依赖包

## 通过Cmake内置模块引入依赖包
为了方便我们在项目中引入外部依赖包，cmake官方为我们预定义了许多寻找依赖包的Module，他们存储在path_to_your_cmake/share/cmake-<version>/Modules目录下。每个以Find<LibaryName>.cmake命名的文件都可以帮我们找到一个包。我们也可以在官方文档中查看到哪些库官方已经为我们定义好了，我们可以直接使用find_package函数进行引用

[官方文档: Find Mou](https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html)

以curl库为例，假设项目需要引入这个库，从网站中请求网页到本地，我们看到官方已经定义好了FindCURL.cmake。所以我们在CMakeLists.txt中可以直接用find_pakcage进行引用。
```
find_package(CURL)
add_executable(curltest curltest.cc)
if(CURL_FOUND)
    target_include_directories(clib PRIVATE ${CURL_INCLUDE_DIR})
    target_link_libraries(curltest ${CURL_LIBRARY})
else(CURL_FOUND)
    message(FATAL_ERROR ”CURL library not found”)
endif(CURL_FOUND)
```

对于系统预定义的 Find<LibaryName>.cmake 模块，使用方法一般如上例所示。

每一个模块都会定义以下几个变量
```python
# 你可以通过<LibaryName>_FOUND 来判断模块是否被找到，如果没有找到，按照工程的需要关闭 某些特性、给出提醒或者中止编译
<LibaryName>_FOUND
<LibaryName>_INCLUDE_DIR or <LibaryName>_INCLUDES 
<LibaryName>_LIBRARY or <LibaryName>_LIBRARIES
```

## 通过find_package引入非官方的库（该方式只对支持cmake编译安装的库有效）
假设此时我们需要引入glog库来进行日志的记录，我们在Module目录下并没有找到 FindGlog.cmake。所以我们需要自行安装glog库，再进行引用。

安装过程
```python
# clone该项目
git clone https://github.com/google/glog.git 
# 切换到需要的版本 
cd glog
git checkout v0.40  

# 根据官网的指南进行安装
cmake -H. -Bbuild -G "Unix Makefiles"
cmake --build build
cmake --build build --target install
```

此时我们便可以通过与引入curl库一样的方式引入glog库了

```python
find_package(GLOG)
add_executable(glogtest glogtest.cc)
if(GLOG_FOUND)
    # 由于glog在连接时将头文件直接链接到了库里面，所以这里不用显示调用target_include_directories
    target_link_libraries(glogtest glog::glog)
else(GLOG_FOUND)
    message(FATAL_ERROR ”GLOG library not found”)
endif(GLOG_FOUND)
```

## Module模式与Config模式

find_package有两种模式，一种是Module模式，也就是我们引入curl库的方式。另一种叫做Config模式，也就是引入glog库的模式。下面我们来详细介绍着两种方式的运行机制。

在Module模式中，cmake需要找到一个叫做Find<LibraryName>.cmake的文件。这个文件负责找到库所在的路径，为我们的项目引入头文件路径和库文件路径。cmake搜索这个文件的路径有两个，一个是上文提到的cmake安装目录下的share/cmake-<version>/Modules目录，另一个使我们指定的CMAKE_MODULE_PATH的所在目录。

如果Module模式搜索失败，没有找到对应的Find<LibraryName>.cmake文件，则转入Config模式进行搜索。它主要通过<LibraryName>Config.cmake or <lower-case-package-name>-config.cmake这两个文件来引入我们需要的库。以我们刚刚安装的glog库为例，在我们安装之后，它在/usr/local/lib/cmake/glog/目录下生成了glog-config.cmake文件，而/usr/local/lib/cmake/<LibraryName>/正是find_package函数的搜索路径之一。（find_package的搜索路径是一系列的集合，而且在linux，windows，mac上都会有所区别，需要的可以参考官方文档find_package）

由以上的例子可以看到，对于原生支持Cmake编译和安装的库通常会安装Config模式的配置文件到对应目录，这个配置文件直接配置了头文件库文件的路径以及各种cmake变量供find_package使用。而对于非由cmake编译的项目，我们通常会编写一个Find<LibraryName>.cmake，通过脚本来获取头文件、库文件等信息。通常，原生支持cmake的项目库安装时会拷贝一份XXXConfig.cmake到系统目录中，因此在没有显式指定搜索路径时也可以顺利找到。

## 编写自己的Find<LibraryName>.cmake模块
我们在当前目录下新建一个ModuleMode的文件夹，在里面我们编写一个计算两个整数之和的一个简单的函数库。库函数以手工编写Makefile的方式进行安装，库文件安装在/usr/lib目录下，头文件放在/usr/include目录下。其中的Makefile文件如下：
```python
# 1、准备工作，编译方式、目标文件名、依赖库路径的定义。
CC = g++
CFLAGS  := -Wall -O3 -std=c++11 

OBJS = libadd.o #.o文件与.cpp文件同名
LIB = libadd.so # 目标文件名
INCLUDE = ./ # 头文件目录
HEADER = libadd.h # 头文件

all : $(LIB)

# 2. 生成.o文件 
$(OBJS) : libadd.cc
    $(CC) $(CFLAGS) -I ./ -fpic -c $< -o $@

# 3. 生成动态库文件
$(LIB) : $(OBJS)
    rm -f $@
    g++ $(OBJS) -shared -o $@ 
    rm -f $(OBJS)


# 4. 删除中间过程生成的文件 
clean:
    rm -f $(OBJS) $(TARGET) $(LIB)

# 5.安装文件
install:
    cp $(LIB) /usr/lib
    cp $(HEADER) /usr/include
```
编译安装
```
make
sudo make install
```
接下来回到Cmake项目中来，在cmake文件夹下新建一个FindAdd.cmake的文件。我们的目标是找到库的头文件所在目录和共享库文件的所在位置。
```python
# 在指定目录下寻找头文件和动态库文件的位置，可以指定多个目标路径
find_path(ADD_INCLUDE_DIR libadd.h /usr/include/ /usr/local/include ${CMAKE_SOURCE_DIR}/ModuleMode)
find_library(ADD_LIBRARY NAMES add PATHS /usr/lib/add /usr/local/lib/add ${CMAKE_SOURCE_DIR}/ModuleMode)

if (ADD_INCLUDE_DIR AND ADD_LIBRARY)
    set(ADD_FOUND TRUE)
endif (ADD_INCLUDE_DIR AND ADD_LIBRARY)
```

这时我们便可以像引用curl一样引入我们自定义的库了。
在CMakeLists.txt中添加
```python
# 将项目目录下的cmake文件夹加入到CMAKE_MODULE_PATH中，让find_pakcage能够找到我们自定义的函数库
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
add_executable(addtest addtest.cc)
find_package(ADD)
if(ADD_FOUND)
    target_include_directories(addtest PRIVATE ${ADD_INCLUDE_DIR})
    target_link_libraries(addtest ${ADD_LIBRARY})
else(ADD_FOUND)
    message(FATAL_ERROR "ADD library not found")
endif(ADD_FOUND)
```