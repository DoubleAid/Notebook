### <font color="deepskyblue">介绍 ROS Launch file</font>
上一章 所用的parameter server， 再关掉 master node 之后， 设置的各种 parameter 就消失了，为了防止每次都需要设置 parameter ，可以通过 launch file

首先，launch file 的文件需要放到 launch 资料夹内
```
cd beginner_toturials
mkdir launch
cd launch
vim startup.launch
```
startup.launch 使用的是xml格式
```xml
<launch>
    <param name="/print_frq" type="double" value="2.0" />
</launch>
```

运行 launch
```
roslaunch [package] [filename.launch]
```
即 运行 startup.launch
```
roslaunch beginner_tutorials startup.launch
```

launch file 不仅可以设置参数，还可以设定一些一开始就需要在背景执行的node，比如
```xml
<launch>
    <param name="/print_frq" type="double" value="2.0" />
    <node name="talker" pkg="beginner_tutorials" type="talker.py">
</launch>
```

另外，通常在执行 node 之前 都需要执行 roscore, 但在 roalaunch 执行的时候， 他会自己检测是否开启了 master， 如果没有就会自动开启