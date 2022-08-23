# 1. 查看环境变量
查看环境变量有三种命令
+ env: env命令是environment的缩写，用于列出所有的环境变量；
+ export：单独使用export命令也可以像env列出所有的环境变量，不过export命令还有其他额外的功能；
+ echo $PATH： echo $PATH用于列出变量PATH的值，里面包含了已添加的目录。

# 2. 设置环境变量
### 1. 把你的路径加入到path  
可以直接添加到环境变量PATH中。$PATH表示变量PATH的值，包含已有的目录。
这种方法需要注意路径的顺序，如果遇到有同名的命令，那么PATH里面哪个目录先被查询，则那个目录下的命令就会被先执行，如下所示：
    
```
# 加到PATH末尾
export PATH=$PATH:/path/to/your/dir

# 加到PATH开头
export PATH=/path/to/your/dir:$PATH
```

### 2. 命名一个新的环境变量

也可以直接命名一个新的环境变量，用于其它程序引用：
```
export VAR_NAME=value
```

### 3. 作用域
环境变量的作用通常有三个
+ 用于当前终端
打开一个终端， 输入添加环境变量的语句
```
export CLASS_PATH=./JAVA_HOME/lib:$JAVA_HOME/jre/lib
```
终端所添加的环境变量是临时的，只适用于当前终端，关闭当前终端或在另一个终端中，添加的环境变量无效。

+ 用于当前用户
如果只需要添加的环境变量对当前用户有效，可以写入用户主目录下的.bashrc文件：
```
vim ~/.bashrc
```
添加语句：
```
export CLASS_PATH=./JAVA_HOME/lib:$JAVA_HOME/jre/lib
```
注销或者重启可以使修改生效，如果要使添加的环境变量马上生效：
```
source ~/.bashrc
```

+ 用于所有用户

要使环境变量对所有用户有效，可以修改profile文件：
```
sudo vim /etc/profile 
```
添加语句：
```
export CLASS_PATH=./JAVA_HOME/lib:$JAVA_HOME/jre/lib
```
注销或者重启可以使修改生效，如果要使添加的环境变量马上生效：
```
source /etc/profile
```