rosbag 是用来记录的工具， 可以几轮运行时的 某个 topic 输出的信息

首先运行 写好的 startup.launch
```
roslaunch beginner_tutorials startup.launch
```

使用 rosbag 记录 /chatter 接收或者发送的讯息
```
rosbag record <topic_name>
```
可以在 运行一段时间后关掉， 就会有一个 bag 文件， 这时候就可以用 rosbag 命令查看 bag 的相关信息

### <font color="deepskyblue">rosbag常见命令</font>

#### <font color="coral">rosbag info</font>
```
rosbag info <bag_name>
```

#### <font color="coral">rosbag play <bag_name></font>
可以使用 rosbag play 重新执行这些记录的message， 并 使用 rostopic echo 显示该 topic 重新接受该包 发布的信息