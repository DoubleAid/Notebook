## 介绍
Protocol Buffer（简称Protobuf），是谷歌出品的序列化框架。与开发语言无关，和平台也无关，具有良好的可扩展性

Protobuf支持生成代码的语言包括 Java，C++，Go，Ruby，C#

[官网地址](https://developers.google.com/protocol-buffers/)

Protobuf的序列化结果体积要比 xml、json 小很多， Protobuf使用了Varint编码，减少数据对空间的占用

Protobuf序列化和反序列化速度比 xml、json 快很多

## 安装
[下载地址](https://developers.google.com/protocol-buffers/)

### ubuntu 安装
```cpp
// 下载 protobuf
$ git clone https://github.com/protocolbuffers/protobuf.git

// 安装依赖库
$ sudo apt-get install autoconf automake libtool curl make g++ unzip libffi-dev -y

// 进入目录
$ cd protobuf

// 自动生成 configure 配置文件
$ ./autogen.sh

// 配置环境
$ ./configure

// 编译源代码
$ make

// 安装
$ sudo make install

// 刷新共享库
$ sudo ldconfig

// 测试
$ protoc -h
```

### ProtoBuf消息定义
ProtoBuf 的消息是在idl文件(.proto)中描述的。

下面是本次样例中使用到的消息描述符customer.proto：

```python
# syntax = "proto3"用于idl语法版本，目前有两个版本proto2和proto3，两个版本语法不兼容，如果不指定，默认语法是proto2。由于proto3比proto2支持的语言更多，语法更简洁，本文使用的是proto3。
syntax = "proto3";

# 其次有一个package domain;定义。此配置用于嵌套生成的类/对象。
package domain;

# 有一个option java_package定义。生成器还使用此配置来嵌套生成的源。此处的区别在于这仅适用于Java。在使用Java创建代码和使用JavaScript创建代码时，使用了两种配置来使生成器的行为有所不同。也就是说，Java类是在包com.protobuf.generated.domain下创建的，而JavaScript对象是在包domain下创建的。
option java_package = "com.protobuf.generated.domain";
option java_outer_classname = "CustomerProtos";


message Customers {
    repeated Customer customer = 1;
}
 

message Customer {
    int32 id = 1;
    string firstName = 2;
    string lastName = 3;
 
    enum EmailType {
        PRIVATE = 0;
        PROFESSIONAL = 1;
    }
 
    message EmailAddress {
        string email = 1;
        EmailType type = 2;
    }
 
    repeated EmailAddress email = 5;
}## 介绍
Protocol Buffer（简称Protobuf），是谷歌出品的序列化框架。与开发语言无关，和平台也无关，具有良好的可扩展性

Protobuf支持生成代码的语言包括 Java，C++，Go，Ruby，C#

[官网地址](https://developers.google.com/protocol-buffers/)

Protobuf的序列化结果体积要比 xml、json 小很多， Protobuf使用了Varint编码，减少数据对空间的占用

Protobuf序列化和反序列化速度比 xml、json 快很多

## 安装
[下载地址](https://developers.google.com/protocol-buffers/)

### ubuntu 安装
```cpp
// 下载 protobuf
$ git clone https://github.com/protocolbuffers/protobuf.git

// 安装依赖库
$ sudo apt-get install autoconf automake libtool curl make g++ unzip libffi-dev -y

// 进入目录
$ cd protobuf

// 自动生成 configure 配置文件
$ ./autogen.sh

// 配置环境
$ ./configure

// 编译源代码
$ make

// 安装
$ sudo make install

// 刷新共享库
$ sudo ldconfig

// 测试
$ protoc -h
```

### ProtoBuf消息定义
ProtoBuf 的消息是在idl文件(.proto)中描述的。

下面是本次样例中使用到的消息描述符customer.proto：

```python
# syntax = "proto3"用于idl语法版本，目前有两个版本proto2和proto3，两个版本语法不兼容，如果不指定，默认语法是proto2。由于proto3比proto2支持的语言更多，语法更简洁，本文使用的是proto3。
syntax = "proto3";

# 其次有一个package domain;定义。此配置用于嵌套生成的类/对象。
package domain;

# 有一个option java_package定义。生成器还使用此配置来嵌套生成的源。此处的区别在于这仅适用于Java。在使用Java创建代码和使用JavaScript创建代码时，使用了两种配置来使生成器的行为有所不同。也就是说，Java类是在包com.protobuf.generated.domain下创建的，而JavaScript对象是在包domain下创建的。
option java_package = "com.protobuf.generated.domain";
option java_outer_classname = "CustomerProtos";


message Customers {
    repeated Customer customer = 1;
}
 

message Customer {
    int32 id = 1;
    string firstName = 2;
    string lastName = 3;
 
    enum EmailType {
        PRIVATE = 0;
        PROFESSIONAL = 1;
    }
 
    message EmailAddress {
        string email = 1;
        EmailType type = 2;
    }
 
    repeated EmailAddress email = 5;
}
```
**关键字解释**
+ **package**： 包名-命名空间
  
  package ourproject.lyphone;该包名在生成对应的C++文件时，将被替换为名字空间名称，既
  ```
  namespace ourproject { 
      namespace lyphone {
          ...
      }
    }
  
  ```
  而在生成的Java代码文件中将成为Java代码文件的包名。
+ **java_package**： 制定生成java类的包名
+ **java_outer_classname**：指定生成Java代码的外部类名称。如果没有指定该选项，Java代码的外部类名称为当前文件的文件名部分，同时还要将文件名转换为驼峰格式，如：my_project.proto
+ **message**： 是消息定义的关键字，等同于C++中的struct/class
+ **字段修饰符**
  + **required**：字段必须提供，否则消息将被认为是 “未初始化的 (uninitialized)”。
  + **optional**：字段可以设置也可以不设置。如果可选的字段值没有设置，则将使用默认值。
  + **repeated**：字段可以重复任意多次 (包括0)。在 protocol buffer 中，重复值的顺序将被保留。将重复字段想象为动态大小的数组。

### 代码生成
```cpp
protoc --proto_path=src_out=build/gen src/foo.proto src/bar/baz.proto
// 编译器读取文件src/foo.proto和src/bar/baz.proto并产生4个输出文件：build/gen/foo.pb.h、build/gen/foo.pb.cc、build/gen/bar/baz.pb.h和build/gen/bar/baz.pb.cc。需要的话，编译器会自动生成build/gen/bar目录，但是并不会创建build或build/gen，因此，它们必须已存在。

protoc -I . --cpp_out=. ./common/proto/**/*.proto
```
+ 使用`--cpp_out=`命令行参数，Protocol Buffer编译器会生成C++输出。
+ `--cpp_out=`选项的参数是你要存放C++输出的目录。
+ 编译器会为每个.proto文件生成一个头文件和实现文件。输出文件的名称与给定的.proto文件名称有关：
  + 后缀（.proto）被替换成.pb.h（头文件）或pb.cc（实现文件）。
  + proto路径（通过--proto_path或-I指定）被输出路径（通过--cpp_out指定）替换。

### 汇总问题
+ base_map.proto:10:1: Import "common/proto/math/geometry.proto" was not found or had errors.
  引用的路径错误，需要会退到 import 的根目录下执行 protoc 命令


### 参考链接
[维基百科](https://en.wikipedia.org/wiki/Protocol_Buffers)

[序列化与反序列化](https://tech.meituan.com/2015/02/26/serialization-vs-deserialization.html)

[官方Benchmark](https://code.google.com/archive/p/thrift-protobuf-compare/wikis/Benchmarking.wiki)

[Charles Protocol Buffers](https://www.charlesproxy.com/documentation/using-charles/protocol-buffers/)

[choose-protocol-buffers](https://codeclimate.com/blog/choose-protocol-buffers/)

[知乎](https://zhuanlan.zhihu.com/p/160249058)

[CSDN](https://blog.csdn.net/shimazhuge/article/details/73825280)

[定义Protobuf消息](https://blog.csdn.net/shimazhuge/article/details/73825280)

[Proto3-C++代码生成指南](https://www.cnblogs.com/lianshuiwuyi/p/12291208.html)

 包名-命名空间
  
  package ourproject.lyphone;该包名在生成对应的C++文件时，将被替换为名字空间名称，既
  ```
  namespace ourproject { 
      namespace lyphone {
          ...
      }
    }
  
  ```
  而在生成的Java代码文件中将成为Java代码文件的包名。
+ **java_package**： 制定生成java类的包名
+ **java_outer_classname**：指定生成Java代码的外部类名称。如果没有指定该选项，Java代码的外部类名称为当前文件的文件名部分，同时还要将文件名转换为驼峰格式，如：my_project.proto
+ **message**： 是消息定义的关键字，等同于C++中的struct/class
+ **字段修饰符**
  + **required**：字段必须提供，否则消息将被认为是 “未初始化的 (uninitialized)”。
  + **optional**：字段可以设置也可以不设置。如果可选的字段值没有设置，则将使用默认值。
  + **repeated**：字段可以重复任意多次 (包括0)。在 protocol buffer 中，重复值的顺序将被保留。将重复字段想象为动态大小的数组。

### 代码生成
```cpp
protoc --proto_path=src_out=build/gen src/foo.proto src/bar/baz.proto
// 编译器读取文件src/foo.proto和src/bar/baz.proto并产生4个输出文件：build/gen/foo.pb.h、build/gen/foo.pb.cc、build/gen/bar/baz.pb.h和build/gen/bar/baz.pb.cc。需要的话，编译器会自动生成build/gen/bar目录，但是并不会创建build或build/gen，因此，它们必须已存在。

protoc -I . --cpp_out=. ./common/proto/**/*.proto
```
+ 使用`--cpp_out=`命令行参数，Protocol Buffer编译器会生成C++输出。
+ `--cpp_out=`选项的参数是你要存放C++输出的目录。
+ 编译器会为每个.proto文件生成一个头文件和实现文件。输出文件的名称与给定的.proto文件名称有关：
  + 后缀（.proto）被替换成.pb.h（头文件）或pb.cc（实现文件）。
  + proto路径（通过--proto_path或-I指定）被输出路径（通过--cpp_out指定）替换。

### 汇总问题
+ base_map.proto:10:1: Import "common/proto/math/geometry.proto" was not found or had errors.
  引用的路径错误，需要会退到 import 的根目录下执行 protoc 命令


### 参考链接
[维基百科](https://en.wikipedia.org/wiki/Protocol_Buffers)

[序列化与反序列化](https://tech.meituan.com/2015/02/26/serialization-vs-deserialization.html)

[官方Benchmark](https://code.google.com/archive/p/thrift-protobuf-compare/wikis/Benchmarking.wiki)

[Charles Protocol Buffers](https://www.charlesproxy.com/documentation/using-charles/protocol-buffers/)

[choose-protocol-buffers](https://codeclimate.com/blog/choose-protocol-buffers/)

[知乎](https://zhuanlan.zhihu.com/p/160249058)

[CSDN](https://blog.csdn.net/shimazhuge/article/details/73825280)

[定义Protobuf消息](https://blog.csdn.net/shimazhuge/article/details/73825280)

[Proto3-C++代码生成指南](https://www.cnblogs.com/lianshuiwuyi/p/12291208.html)

