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

解决方案：
+ https://github.com/deu/palemoon-overlay/issues/31.
+ https://superuser.com/questions/112834/how-to-match-whitespace-in-sed
+ https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=850984
+ https://github.com/MoonchildProductions/Pale-Moon/issues/872
+ https://bug1329272.bmoattachments.org/attachment.cgi?id=8825307