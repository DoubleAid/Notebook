### <font color="deepskyblue">Service 介绍</font>
与 topic的 publisher 和 subscriber 不同
Service更像是 server 和 client之间的交互， 两者通过 Service 进行沟通， 也就是client 发送 request 给 server， server 再传回 response 给 client

Service 只有一个 server， 只能 一个 server 对应 多个 client。

### <font color="deepskyblue">使用python编写server node</font>
建立一个名为 add_two_ints_server.py, 内容如下：
```python
import rospy
from rospy_tutorials.srv import *

def handle_add_two_ints(req):
    print("Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b)))
    return AddTwoIntsResponse(req.a + req.b)

def add_two_ints_server():
    rospy.init_node("add_two_ints_server")
    # Service 的定义如下
    # rospy.Service(name, service_class, handler, buff_size=65536)
    # 第一个参数 为 Service 的名称， 接着是 service 的格式， 这是官方定义好的 Service 的格式， 也就是 AddTwoInts， 第三个参数是 呼叫这个 service 的时候 callback， 第四个buff_size是指接收的request的最大长度
    s = rospy.Service("add_two_ints", AddTwoInts, handle_add_two_ints)
    print("Ready to add two ints")
    rospy.spin()


if __name__ == "__main__":
    add_two_ints_server()
```
只要运行这个service server node就可以启用 server了
为了测试，可以通过 service call 
```
rosservice call /add_two_ints 5 6
>>> sum: 11
```

### <font color="deepskyblue">使用python编写client node</font>
```python
import sys
import rospy
from rospy_tutorials.srv import *

def add_two_ints_client(x, y):
    rospy.wait_for_service("add_two_ints")
    try:
        add_two_ints = rospy.ServiceProxy("add_two_ints", AddTwoInts)
        resp1 = add_two_ints(x, y)
        return resp1.sum
    except rospy.ServiceException, e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [x y]"%sys.argv[0]


if __name__ == "__main__":
    if len(sys.argv) == 3:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
    else:
        print(usage())
        sys.exit(-1)
    print("Requesting %s+%s"%(x, y))
    print("%s + %s = %s"%(x, y, add_two_ints_client(x, y)))
```

client 并不需要 init_node(), 直接调用 service 就可以了
```
rospy.wait_for_service(service_name, timeout=None)
```
表示 需要监听 service，只有 service 已经开启，才会继续执行接下来的指令
第一个参数为service的名称， 第二个参数为等待的时间，如果超时就会传回一个 exception 表示发生了异常

```
rospy.ServiceProxy(service_name, service_class, persistent=False, headers=None)
```
+ 第一个参数 表示 service的名称
+ 第二个参数 表示 service的格式
+ 第三个参数 表示是否要让这个client一直和server连接， 这个设定可以允许client在呼叫service的时候再去连接别的node，但需要定义连接失败时该怎么处理
+ 第四个参数 是在每次 调用 service 时可以放在头部的参数， 可以用来制作cookies

写完后在 根目录先 `catkin_make`, 以确保 service 的建立

### <font color="deepskyblue">使用C++编写Server Node</font>
参考链接： https://ithelp.ithome.com.tw/articles/10207224
### <font color="deepskyblue">使用C++编写Client Node</font>
参考链接： https://ithelp.ithome.com.tw/articles/10207252

### <font color="deepskyblue">Service_class的格式</font>
service_class 本质上就是结构体
其结构定义如下,具体的可参考以上的具体用例
```cpp
namespace package_name {
struct service_name {
    class Request {
        ...
    }
    class Response {
        ...
    }
}
}
```