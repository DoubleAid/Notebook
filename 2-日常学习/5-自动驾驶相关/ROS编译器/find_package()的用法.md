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