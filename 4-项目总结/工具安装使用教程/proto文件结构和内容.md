以一个 proto 文件为例
```proto
syntax = "proto3";
package addressbook;

message Person {
    required string name = 1;
    required int32 id = 2;
    optional string email = 3;

    enum PhoneType {
        MOBILE = 0;
        HOME = 1;
        WORK = 2;
    }

    message PhoneNumber {
        required string number = 1;
        optional PhoneType type = 2 [default = HOME];
    }

    repeated PhoneNumber phone = 4;
}

message AddressBook {
    repeated Person person_info = 1;
}
```
### 代码解释
+ syntax="proto2"; 表明使用protobuf的编译器版本为v2，目前最新的版本为v3
+ package addressbook; 声明了一个包名，用来防止不同的消息类型命名冲突，类似于 namespace 
+ import "src/help.proto";  导入了一个外部proto文件中的定义，类似于C++中的 include 。不过好像只能import当前目录及当前目录的子目录中的proto文件，比如import父目录中的文件时编译会报错（Import "../xxxx.proto" was not found or had errors.），使用绝对路径也不行，尚不清楚原因，官方文档说使用 -I=PATH 或者 --proto_path=PATH 来指定import目录
+ message 是Protobuf中的结构化数据，类似于C++中的类，可以在其中定义需要处理的数据  
+ required string name = 1; 声明了一个名为name，数据类型为string的required字段，字段的标识号为1
  protobuf一共有三个字段修饰符：  
        - required：该值是必须要设置的；  
        - optional ：该字段可以有0个或1个值（不超过1个）；  
        - repeated：该字段可以重复任意多次（包括0次），类似于C++中的list； 
+ optional PhoneType type = 2 [default = HOME]; 为type字段指定了一个默认值，当没有为type设值时，其值为HOME。另外，一个proto文件中可以声明多个message，在编译的时候他们会被编译成为不同的类。