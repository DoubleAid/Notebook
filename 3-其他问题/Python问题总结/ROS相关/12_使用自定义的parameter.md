 + ### <font color="deepskyblue">in python</font> 
    在python中 就是调用 rospy的get_param
    ```
    rospy.get_param("/param_name", "default_value")
    ```
    第一个参数 为设定的parameter的名称， 第二个参数是在找不到设定的参数时的默认值
    ```python
    import rospy

    rospy.init_node("hello_python_node")
    frq = rospy.get_param("/print_frq")

    while not rospy.is_shutdown():
        rospy.loginfo("Hello World")
        rospy.sleep(frq)
    ```

    python 不仅仅可以使用 get_param, 一样的可以使用 set 或者 delete

 + ### <font color="deepskyblue">in C++</font>
    C++ 比较麻烦一点
    ```cpp
    ros::NodeHandle nh;
    int param;
    nh.getParam("/param_name", param);
    ```
    需要注意的是， getParam() 的用法第一个获取的值放入第二个参数值中
    ```cpp
    #include <ros/ros.h>

    int main(int argc, char** argv) {
        ros::init(argc, argv, "hello_cpp_node");
        ros::NodeHandle handler;

        double frq;
        handler.getParam("/print_frq", frq);

        while (ros::ok()) {
            ROS_INFO("hello world");
            ros::Duration(frq).sleep();
        }
        return 0;
    }
    ```