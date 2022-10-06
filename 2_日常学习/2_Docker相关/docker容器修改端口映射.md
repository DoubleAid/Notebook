[参考链接](https://zhuanlan.zhihu.com/p/94949253)

- [一-通过修改配置文件修改docker容器的端口映射](#一-通过修改配置文件修改docker容器的端口映射)

# 一： 通过修改配置文件修改docker容器的端口映射
1. 使用`docker ps -a` 命令找到要修改容器的 ContainerID
2. 查看 容器的 id
```
# 通过以下命令 获得 container 的本地 Id
docker inspect CONTAINER_ID | grep Id

# 进入 容器文件夹
cd /var/lib/docker/containers
```
3. 停止容器和主机的docker服务
```
# 停止容器
docker stop CONTAINER_ID
# 停止主机docker服务
systemctl stop docker
```
4. 进入 2 中得到的文件夹内， 修改 hostconfig.json 和 config.v2.json
修改 `hostconfig.json`
```
# 比如新增一个 80 端口， 在 PortBindings 下边添加以下内容，端口配置之间适应逗号隔开
"80/tcp": [
    "HostIp": "0.0.0.0",
    "HostPort": "80"
]
```
修改 config.v2.json, 找到 `ExposedPorts` 和 `Ports`, 仿照之前内容添加端口映射
```
"ExposePorts": {
    "2000/tcp": {}
},

"Ports": {
    "2000/tcp": [
        {
            "HostIp": "0.0.0.0",
            "HostPort": "2000"
        }
    ]
},
```
5. 保存之后重启 docker 服务和容器
```
systemctl start docker
docker start CONTAINER_ID
```

# 二： 把运行中的容器生成新的镜像， 再新建容器
1. 提交一个运行中的容器为镜像
```
docker commit CONTAINER_ID IMAGE_NAME
```
2. 运行新建的镜像并添加端口映射
```
docker run -d -p 8000:80 IMAGE_NAME /bin/bash
```