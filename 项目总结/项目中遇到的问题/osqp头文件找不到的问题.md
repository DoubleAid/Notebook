### 问题描述
系统中添加了 osqp 库之后，catkin_make 报错无法找到osqp的头文件，拷贝相应的 头文件 放在 ${工程目录}/src/third_party 下仍无法解决

### 解决方法
从 oss://allride-release/third_party/ 中下载所有的osqp文件（推荐） 或者 自己编译相应的文件
拷贝文件至/opt/allride/third_party/ 下，文件夹需要创建

