# rosnode的常见指令

##  <font color="deepskyblue">rosrun 帮助找到并执行node</font> 

命令格式

```bash
$ rosrun <package> <executable>
```

第一个参数为 package 的名称， 第二个为可执行文件的名称
如：

```bash
$ rosrun beginner_tutorials hello.py 
```

## <font color="deepskyblue">rosnode list 列出运行中的node</font> 

## <font color="deepskyblue">rosnode info <node_name></font> 

这个指令旨在查看某个node的信息， 包括 publications 和 subscriptions 等等，这些之后会讲的

## <font color="deepskyblue">rosnode ping 检查特定node的运行情况</font> 

指令格式

```cpp
$ rosnode ping <node_name>
```

这个指令和常用的ping ip地址类似，会传回运行时的时间戳，主要用于查看这个node有没有运行成功

### <font color="deepskyblue">rosnode kill 中止特定的 node</font> 

```cpp
rosnode kill <node_name>
```
