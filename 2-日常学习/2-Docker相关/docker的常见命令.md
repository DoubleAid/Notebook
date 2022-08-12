- [register](#register)
- [运行数据集](#运行数据集)
- [docker run 参数解释](#docker-run-参数解释)
  - [例子](#例子)
- [docker build 参数解释](#docker-build-参数解释)

### register 
```
sudo docker login --username=guanggang.bian@allride allride-registry.cn-shanghai.cr.aliyuncs.com
```
如果之前运行docker都用了sudo，就一直用sudo，没有用就不要用

### 运行数据集
```
sudo docker run allride-registry.cn-shanghai.cr.aliyuncs.com/map/map_storage:v2_8.0.100

查看 ip 地址
sudo docker ps
sudo docker inspect --format='{{.NetworkSettings.IPAddress}}' xxxxxxx(相应的id)
```

### docker run 参数解释
+ <font color=deepskyblue>-v, --volume=[]</font> : 用于将 容器内的内容留在本地上，可存放在主機檔案系統中的任何地方，非 Docker 行程或 Docker 容器可隨時修改。
+ <font color=deepskyblue>-idt</font> : idt 是三个命令， 通常在一起使用， -i 以交互模式运行容器， -t 为容器重新分配一个伪终端， -d 后台运行容器， 并返回容器的id

#### 例子
使用docker镜像nginx:latest以后台模式启动一个容器,并将容器命名为mynginx。
```
docker run --name mynginx -d nginx:latest
```

使用镜像nginx:latest以后台模式启动一个容器,并将容器的80端口映射到主机随机端口。
```
docker run -P -d nginx:latest
```

使用镜像 nginx:latest，以后台模式启动一个容器,将容器的 80 端口映射到主机的 80 端口,主机的目录 /data 映射到容器的 /data
```
docker run -p 80:80 -v /data:/data -d nginx:latest
```

绑定容器的 8080 端口，并将其映射到本地主机 127.0.0.1 的 80 端口上。
```
$ docker run -p 127.0.0.1:80:8080/tcp ubuntu bash
```

使用镜像nginx:latest以交互模式启动一个容器,在容器内执行/bin/bash命令
```
docker run -it nginx:latest /bin/bash
```

运行
```
docker run -idt -v /var/run/docker.sock -v /home/local/path/:/container/path/app --net=host image-name:tag
```

本地打开 docker 终端
```
docker exec -it container_id/container_name bash
```

规定接口运行
```
docker run -idt -p 5432:5432 {container_id/container_name}
```
### docker build 参数解释
+ --rm 设置镜像成功后删除中间容器

参考运行方式
```
docker build --rm -f Dockerfile_service --build-arg AA=aa --build-arg BB=bb -t {build_container_name:tag} .
```

