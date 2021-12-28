首先要先去修改 beginner_tutorials 里面的 package.xml 和 CMakeList.txt 的设定

#### <font color="coral">in package.xml</font>
新增两行代码，一个是建立 message 用， 另一个 执行的时候用
```xml
<build_depend>message_generation</build_depend>
<exec_depend>message_runtime</exec_depend>
```

#### <font color="coral">in CMakeLists.txt</font>
新增建立message用的依赖模块， 在 find_package 底下新增一个 message_generation
```
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation
)
```
并且在 catkin_package 底下将 CATKIN_DEPEND 的注释打开， 在后面新增一个 message_runtime
```
catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES beginner_tutorials
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
  # DEPENDS system_lib
  )
```
其他部分只需要把注释取消掉即可
```
generate_message (
    DEPENDENCIES
    std_msgs
)
```

以上就完成 message 的基本设定

#### <font color="coral">定义自己的message</font>
ROS 建立的message都需要放置在 /msg 的文件夹内， 所以需要自己创建 msg文件夹
```
cd beginner_tutorials
mkdir msg
cd msg
vim my_msg.msg
```
以一个文章为例，需要 id、title、content
```
int64 id
string title
string content
```
编写好之后回去继续编写 CMakeList.txt, 修改 add_message_files
```
add_message_files(
    FILES
    my_msg.msg
)
```
这样就完成了设定，只需要到根目录下执行catkin_make 就可以建立不同形式的 my_msg了

