首先看一下几个常用的指令
#### <font color="coral">rosmsg list</font>
查看所有可以使用的message

#### <font color="coral">rosmsg package <package_name></font>
显示某个package下的所有message

#### <font color="coral">rosmsg show <package_name>/<msg_name></font>
显示某个 message 的具体信息

<font color="red">运行之前需要先 source 一下 devel/setup.bash</font>

#### <font color="coral">在 C++ 中使用自定义的message</font>
```cpp
#include <beginner_tutorials.my_msg.h>
beginner_tutorials::my_msg msg;
```

+ talker.cpp
    ```cpp
    
    ```
#### <font color="coral">在 python 中使用自定义的message</font>
```python
from beginner_tutorials.msg import my_msg
msg = my_msg()
```
+ talker.py
    ```python
    import rospy
    from beginner_tutorials.msg import my_msg

    def talker():
        pub = rospy.Publisher("chatter", my_msg, queue_size=10)
        rospy.init_node("talker", anonymous=True)
        rate = rospy.Rate(10)
        count = 1
        while not rospy.is_shutdown():
            msg = my_msg()
            msg.id = count
            msg.title = "hello"
            msg.content = "hello from python"
            pub.publish(msg)
            count = count + 1
            rate.sleep()


    if __name__ == "__main__":
        try:
            talker()
        except rospy.ROSInterruptException:
            pass
    ```