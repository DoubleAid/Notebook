编写的 srv 文件就是自定义service的消息格式，编写的方式和之前的 .msg 类似
```
cd beginner_tutorials
mkdir srv
cd srv
vim my_srv.srv
```
在 my_srv.srv 中编写
```
int64 id
---
string name
string gender
int64 age
```
意思为 id 与下面的信息相对应， 输入某个人的id， 就可以知道他的姓名、性别和年龄
#### <font color="coral">修改 CMakeList.txt</font>
修改 add_service_files
```
add_service_files(
    FILES
    my_srv.srv
)
```
设定完之后回到根目录执行 `catkin_make` , 自动建立service