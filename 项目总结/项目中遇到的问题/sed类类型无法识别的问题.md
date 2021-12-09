### 问题
```
sed: character class syntax is [[:space:]], not [:space:]
checking for correct C++ linkage of basic libpq functions... configure: error: 
Linking a call to libpq failed in C++, even though it succeeded in C.  If your
C and C++ compilers are very different beasts, this may mean that we do not have
the right options for linking with it after all.
```

代码自带的配置文件在编译过程中，会去调用sed4.3版本，而主机系统的sed是4.5版本，版本会不一致。编译的时候，sed命令修改文件就识别不了空格键。

sed4.3版本以前的空格键：[:space:]
sed4.3版本以后的空格键：[[:space:]]

### 解决方案：
安装 sed4.2
下载地址： https://ftp.gnu.org/gnu/sed/
具体安装过程参考 安装包内的 readme