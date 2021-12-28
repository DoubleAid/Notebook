### <font color="deepskyblue">rosservice list</font>
```
$ rosservice list
```
显示 运行中的 service

### <font color="deepskyblue">rosservice info</font>
```
$ rosservice info <service_name>
```
查看某个特定service的相关信息

### <font color="deepskyblue">rosservice call</font>
```
$ rosservice call <service_name> [argv1 argv2 ...]
```
就是调用 某个 service， 并传递参数