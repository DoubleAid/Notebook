### ROS介绍
ROS 全名为 Robot Operating System， 顾名思义是 机器人操作系统， 但和平常的操作系统并不相同， ROS是安装在linux的环境上的，ROS主要是负责机器人的各个原件进行沟通与操作的一个框架

### Package
Package 即为模组/包， 可以引用别人的模组或者自己编写出来，可以通过catkin工具在workspace中建立package
```shell
catkin_create_pkg <package_name> [depend1] [depend2] [depend3]

# 例如
cd catkin_ws/src
catkin_create_pkg beginner_tutorials std_msgs rospy roscpp
```
### ROS工程搭建
+ 先新建文件夹 假设 工程名称为 catkin_ws, `mkdir -r catkin_ws/src`
+ 进入 catkin_ws ,建立工程
    ```
    cd catkin_ws
    catkin_make
    ```
+ 创建package
    ```shell
    # std_msgs、rospy、roscpp 为 package的依赖
    catkin_create_pkg beginner_tutorials std_msgs rospy roscpp
    ```

### Nodes
Node是 ROS系统内最小的一个单位，一个package包含有许多的node用来执行不同的功能
每个Node都可以和别的Node通过 Topic 或 Service 或者 其他方式来沟通，获取数据
Node 的一个很大的特点就是可以使用不同的语言来进行编写， 以人物侦测为例， 获取图片及影像处理的部分可以用 Python 来写，特征点侦测比较注重性能，可以改用C++来写，这些不同的Node都可以放在同一个Package里面，形成一个人物侦测的模组


### 使用 python 写 ros node
```python
import rospy
rospy.init_node("hello_python_node")
rospy.loginfo("hello world")
```
这样一个简单的 node 就编写完成
但在执行时会遇到以下的报错
```
Unable to register with master node [http://localhost:11311]: master may not be running yet. Will keep trying
```
这是因为 node 需要一个 master(主人)去管理， 因此需要先打开 master 才能执行 node

### Master
master 就是ROS系统中负责管理 Node 的进程，负责 node 之间的沟通，因此在执行node之前需要先把 master 开启，
```shell
$ roscore
```

接着可以修改 python 文档，让其每秒打印一次hello world
```python
import rospy

rospy.init_node("hello_python_node")

while not rospy.is_shutdown():
    rospy.loginfo("Hello World")
    rospy.sleep(1)
```

在node执行期间，master 可以通过 rosnode 的相关指令， 将运行中的 node 找出来, 例如 rosnode list 就可以显示正在运行的 node

### 使用 C++ 编写 ROS Node
首先和python一样，编写 cpp 文件
```cpp
#include <ros/ros.h>
using namespace std;

int main() {
    ros::init(argc, argv, "hello_cpp_node");
    ros::NodeHandle handler;
    ROS_INFO("hello world");
}
```

可以看出 C++ 与 Python 相比，只是多了一个 handler，是用来和其他的node沟通的 topics，services, parameters等的一个介质

cpp文件需要在编译后才可以执行， 需要修改相应的 CMakeLists.txt, 设置相连接的函数库，可以参考 CMakeLists.txt 的编写

可以将cpp修改，让其每秒执行一次
```cpp
#include <ros/ros.h>
using namespace std;

int main() {
    ros::init(argc, argv, "hello_cpp_node");
    ros::NodeHandle handler;
    while (ros::ok()) {
        ROS_INFO("hello world");
        ros::Duration(1).sleep();
    }
}
```

