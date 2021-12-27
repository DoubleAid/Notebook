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
from std_msg.msg import String

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
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