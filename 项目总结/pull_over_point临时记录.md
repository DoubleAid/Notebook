要处理 pull_over_point, 首先要在本地编译 处理的代码， 再调用相应的工具对 pull_over_point.pb.txt 进行处理
最后更新 grpc 接口
### <font color="deepskyblue">拉取代码到本地</font>
pull_over_point 数据上传的代码库为 https://gitlab.allride-ai.cn/map/map_service/-/tree/grpc
通过 git 拉取代码
```
git clone -b grpc git@gitlab.allride-ai.cn:map/map_service.git
```

### <font color="deepskyblue">搭建 docker 容器</font> 
```
sudo docker run -idt -v var/run/docker.sock -v /home/guanggang.bian/Private/Code/C++/map_service/:/app -net=host allride-registry-cn-shanghai.cr.aliyuncs.com/map/map.grpc:v3.0.1
```

### <font color="deepskyblue">进入 容器</font>
```
sudo docker exec -it {container id} bash
```

### <font color="deepskyblue">copy redis 依赖</font>
```
find / -name libhiredis.so
cd {path to libhiredis}
cp libhiredis.so /usr/local/lib/libhiredis.so.1.0.3-dev
cp libhiredis.so /usr/local/lib/libhiredis.so
```

### <font color="deepskyblue">编译代码</font>
```
./build.sh
# 如果失败，需要指定 conan-local
conan install .. -r conan-local
```

### <font color="deepskyblue">查看 set_data.cc, 将 pull_over_points.pb.txt 放到指定的目录下</font>
```
docker cp /download/pull_over_points.pb.txt {container id}:/app/{target dir}/
```

### <font color="deepskyblue">上传数据</font>
执行命令
```
./devel/lib/server/set_data
```

### <font color="deepskyblue">更新 服务</font>
注意，需要使用 公司的 k8s config
```
kubectl -n ridesharing delete pod render-map-service-v2-856dc576c-l7l2c
```