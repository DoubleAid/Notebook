- [<font color="coral">在文件中插入一行</font>](#font-colorcoral在文件中插入一行font)
- [<font color="coral">删除行</font>](#font-colorcoral删除行font)
- [<font color="coral">搜索 testfile 有 oo 关键字的行</font>](#font-colorcoral搜索-testfile-有-oo-关键字的行font)


### <font color="coral">在文件中插入一行</font>
```
$ cat testfile #查看testfile 中的内容  
HELLO LINUX!  
Linux is a free unix-type opterating system.  
This is a linux testfile!  
Linux test 
Google
```
在 testfile 文件的第四行后添加一行，并将结果输出到标准输出，在命令行提示符下输入如下命令： 
```
sed -e 4a\newLine testfile 
```
```
$ sed -e 4a\newLine testfile 
HELLO LINUX!  
Linux is a free unix-type opterating system.  
This is a linux testfile!  
Linux test 
newLine
Google
```

### <font color="coral">删除行</font>
<font color="skyblue">将 testfile 的内容列出并且列印行号，同时将第 2~5 行删除</font>
```
$ nl testfile | sed '2,5d'
     1  HELLO LINUX!  
     6  Taobao
     7  Runoob
     8  Tesetfile
     9  Wiki
```
ed 的动作为 2,5d，那个 d 是删除的意思，因为删除了 2-5 行，所以显示的数据就没有 2-5 行了， 另外，原本应该是要下达 sed -e 才对，但没有 -e 也是可以的，同时也要注意的是， sed 后面接的动作，请务必以 '...' 两个单引号括住

<font color="skyblue">只删除第 2 行</font>
```
$ nl testfile | sed '2d' 
     1  HELLO LINUX!  
     3  This is a linux testfile!  
     4  Linux test 
     5  Google
```

<font color="skyblue">删除第 3 到最后一行</font>
```
$ nl testfile | sed '3,$d' 
     1  HELLO LINUX!  
     2  Linux is a free unix-type opterating system.  
```

### <font color="coral">搜索 testfile 有 oo 关键字的行</font>
搜索 testfile 有 oo 关键字的行:
```
$ nl testfile | sed -n '/oo/p'
     5  Google
     7  Runoob
```
删除 testfile 所有包含 oo 的行，其他行输出
```
$ nl testfile | sed  '/oo/d'
     1  HELLO LINUX!  
     2  Linux is a free unix-type opterating system.  
     3  This is a linux testfile!  
     4  Linux test 
```
搜索 testfile，找到 oo 对应的行，执行后面花括号中的一组命令，每个命令之间用分号分隔，这里把 oo 替换为 kk，再输出这行：
```
$ nl testfile | sed -n '/oo/{s/oo/kk/;p;q}'  
     5  Gkkgle
```