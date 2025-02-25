# ros topic 的用法

之前已经了解了不同的node实现不同的功能，而 不同 node 之间的沟通需要通过 topic 进行

### <font color="deepskyblue">Topic</font>

topic 有点像 tag 一样的标签功能，如果我 mark 了某个topic， 那么只要有 某个 node 标记的这个 topic， 我就可以收到通知 有这个标签的文章（node）上线了，可以查看node的信息。

node 可以利用发布关于某种 topic 的信息，而如果有 node 正在监听 这个 topic 的信息，就可以接受到这个信息，这就是 publisher（发布者） 和 subscriber（订阅者）

publisher 和 subscriber 之间的关系可以是一对一，一对多，多对多

topic 机制是通过 TCP/IP 来传输的， 可以通过 rosnode info 查看node 的ip地址

### <font color="deepskyblue">Message</font>

两个 node 通过 topic 传递信息时，需要先约定好传递的信息的格式，每个Topic需要先限定使用怎样的message type， 比如 控制动作的 actionlib_msgs、导航用的nav_msgs、传感器用的 sensor_msgs等等，有些别人已经写好了，也可以自己定义 message

### <font color="deepskyblue">用python写Publisher</font>

```python
import rospy
# topic 需要 std_msgs.msg 来定义消息的格式
from std_msgs.msg import String

def talker():
    # 设定 node 和 topic 之间的关系， 函数的定义为
    # rospy.Publisher(topic_name, msg_class, queue_size)
    # topic_name: 即为 消息 topic 的名称，
    # msg_class: 消息的格式
    # queue_size：表示这个 topic 一次可以放几个信息
    pub = rospy.Publisher('chatter', String, queue_size=10)
    # 把这个 node 进行初始化，第二个参数 anonymous 表示是否需要匿名， 如果 anonymous=True，就会在原本的node名称后面加上一个乱码，可以方便一次执行多个node，因为node 的名称不能重复
    rospy.init_node('talker', anonymous=True)
    # 设定间隔的频率，也就是每秒需要发送几次讯息，中间的参数是用 HZ 表示的，10就表示每秒发送10次讯息
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
```

运行这个文件，可以看到 有大量的 INFO 被打印出来，这是由 rospy.loginfo 打印出来的， 就像 rosnode list 查看 node 一样， rostopic list 可以查看 topic的运行状态

可以使用 `rostopic echo /chatter` 查看该topic 发送的信息

### <font color="deepskyblue">用python写Subscriber</font>

```python
import rospy
from std_msgs.msg import String

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def listener():
    # 初始化一个叫做 listener 的 node
    rospy.init_node("listener", anonymous=True)
    # 订阅或者监听 名为 "chatter"的topic，通过 String 格式进行信息传输，并在接收到信息后调用 callback 功能
    rospy.Subscriber("chatter", String, callback)
    # spin 的功能就是让这个 node 保持持续的运转
    rospy.spin()


if __name__ == "__main__":
    listener()
```

### <font color="deepskyblue">使用C++编写Publisher</font>

参考链接：https://ithelp.ithome.com.tw/articles/10205657

```cpp
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>

int main(int argc, char** argv) {
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;
    // 建立 Publisher 的函数主要就是通过 advertise方法实现，其 函数定义为
    // template<class M>
    // ros::Publisher advertise(const std::string& topic, uint32_t queue_size, bool latch=false);
    // class M 表示讯息的格式， 可以有很多种格式
    // advertise 的三个参数， 第一个为 
    ros::Publisher chatter_pub = n.advertise<std::String>("chatter", 1000);
    ros::Rate loop_rate(10);

    int count = 0;
    while (ros::ok()) {
        std::stringstream ss;
        ss << "hello world" << count;
        msg.data = ss.str();

        ROS_INFO("%s", msg.data.c_str());
        chatter_pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
        count++;
    }
    return 0;
}
```

### <font color="deepskyblue">使用C++编写Subscriber</font>

参考链接： https://ithelp.ithome.com.tw/articles/10205877
