### <font color="deepskyblue">parameter 介绍</font>
ros master 下的 parameter server 是用来管理全局参数的， 通过调整全局的参数，实现某一些改变
例如之前写的 hello world， 功能是间隔一秒打印一次 print("hello world"), 但如果现在想要调整为 间隔两秒 或者 10 秒打印一次， 每一次都要修改源码太过麻烦，可以通过parameter server 来解决

### <font color="deepskyblue">parameter 常见命令</font>
#### <font color="coral">rosparam list</font>
查看已经存在的 参数

#### <font color="coral">rosparam set</font>
```
rosparam set <parameter_name> <value>
```
这个指令用于建立新的参数，或者是给某个参数设定一个新的值
比如 我想建立一个调整 打印间隔 的参数
```
rosparam set print_frq 10
```

#### <font color="coral">rosparam get</font>
```
$ rosparam get <parameter_name>
```
这个指令可以看到某个特定的参数里面的值是多少
```
rosparam get /print_frq
>>> 10
```