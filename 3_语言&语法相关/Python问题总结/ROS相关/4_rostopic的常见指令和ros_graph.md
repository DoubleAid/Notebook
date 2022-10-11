### <font color="deepskyblue">rostopic list</font>
```shell
$ rostopic list
```
列出目前正在运行的topic有哪些

### <font color="deepskyblue">rostopic info</font>
```
$ rostopic info <topic_name>
```
显示某个topic的详细资料

### <font color="deepskyblue">rostopic echo</font>
```
$ rostopic echo <topic_name>
```
监听某个topic

### <font color="deepskyblue">rostopic pub</font>
```
$ rostopic pub <topic_name> <topic-type> [data]
```
这个执行与echo是相对的，也就是充当 publisher 的指令，负责对某个 topic 发送讯息, 例如：
```
rostopic pub /chatter std_msgs/String "hello from command line"
```

### <font color="deepskyblue">ROS Graph</font>
讲完 topic， 可以讲一下 Graph， ROS 的资料结构是使用图形（graph）的方式，并且提供一个好用的图形化工具， 可以看出每个 node 的链接状况
```
$ rosrun rqt_graph rqt_graph
$ rqt_graph
```
这两个指令都可以执行