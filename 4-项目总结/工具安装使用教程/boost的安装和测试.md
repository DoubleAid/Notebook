### <font color=skyblue>boost介绍</font>
Boost是为C++语言标准库提供扩展的一些C++程序库的总称。Boost库是一个可移植、提供源代码的C++库，作为标准库的后备，是C++标准化进程的开发引擎之一，是为C++语言标准库提供扩展的一些C++程序库的总称。

参考资料：[维基百科](https://zh.wikipedia.org/wiki/Boost_C%2B%2B_Libraries)

### <font color=skyblue>下载并解压</font>
下载链接： https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/

解压
```
tar zxvf boost_1_58_0.tar.gz

# 进入解压后的文件夹
cd boost_1_58_0
```

### <font color=skyblue>编译</font>
有三种 linux boost 的安装方式

#### <font color=chocolate> 第一种 快捷编译（推荐） </font>
在 Linux 操作系统下安装 Boost 很容易，最简单省事的方法是在 Boost 解压缩后的目录下直接执行以下命令： 
```
./bootstrap.sh
./b2.install
```
第一条命令 bootstrap.sh 是编译前的配置工作，第二条命令才开始真正地编译并安装 Boost。 
如果像上面这样不指定额外选项，Boost 将编译 release 版本的库文件，把头文件安装到“/usr/local/include”中，把库文件安装到“/usr/local/lib”中。 

#### <font color=chocolate> 第二种 完全编译 </font>
我们也可以完整编译 Boost，使用 buildtype 选项指定编译类型（如不指定编译类型则默认使用 release 模式），在执行 bootstrap.sh 命令之后执行如下命令： 
```
./b2 --buildtype=complete install
```
这样将开始对 Boost 进行完整编译，安装所有调试版、发行版的静态库和动态库。 

#### <font color=chocolate> 第三种 定制编译 </font>
完整编译 Boost 费时费力，而且在开发过程中这些库并不会全部用到，因此，Boost 允许用户自行选择用户要编译的库。

查看所有必须编译后才能使用的库。
```
./b2 --show-libraries
```

在完全编译命令的基础上，使用--with或--without选项可打开或关闭某个库的编译，如：
```
./b2 --with date_time --buildtype=complete install
```

执行上述命令将编译安装date_time

### <font color=skyblue>测试</font>
可以执行以下代码测试 boost 安装是否完成
```cpp
#include <iostream>
#include <boost/version.hpp>

using namespace std;

int main() {
    cout << BOOST_VERSION << endl;
    return 0;
}
```
头文件 <boost/version.hpp> 里有两个宏，这两个宏定义了当前使用的 Boost 程序库的版本号：

### 运行不成功
#### <font color=chocolate> 出现 fatal error: boost/numeric/ublas/matrix.hpp: </font>
安装 libboost-all-dev
```
sudo apt-get update
sudo apt-get install libboost-all-dev
```