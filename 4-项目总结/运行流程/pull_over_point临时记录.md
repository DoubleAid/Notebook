要处理 pull_over_point, 首先要在本地编译 处理的代码， 再调用相应的工具对 pull_over_point.pb.txt 进行处理 最后更新 grpc 接口
### 拉取代码到本地

pull_over_point 数据上传的代码库为 https://gitlab.allride-ai.cn/map/map_service/-/tree/grpc 通过 git 拉取代码
```
git clone -b grpc git@gitlab.allride-ai.cn:map/map_service.git
```
### 搭建 docker 容器
```
sudo docker run -idt -v var/run/docker.sock -v /home/guanggang.bian/Private/Code/C++/map_service/:/app --net=host allride-registry-cn
            - name: OSS_ACCESS_KEY_ID
              value: "LTAI5tPRfctVqha9zYMABb23"
            - name: OSS_ACCESS_KEY_SECRET
              value: "Xq3DqSuoLXSzgOmj6Wclh7K4SttUyF"
            - name: ALIYUN_REGION
              value: "cn-shanghai"-shanghai.cr.aliyuncs.com/map/map_grpc:v3.0.1
```
### 进入 容器
```
sudo docker exec -it {container id} bash
```
### copy redis 依赖
```
find / -name libhiredis.so
cd {path to libhiredis}
cp libhiredis.so /usr/local/lib/libhiredis.so.1.0.3-dev
cp libhiredis.so /usr/local/lib/libhiredis.so
```
### 编译代码
```
./build.sh
# 如果失败，需要指定 conan-local
conan install .. -r conan-local
```
### 查看 set_data.cc, 将 pull_over_points.pb.txt 放到指定的目录下
```
docker cp /download/pull_over_points.pb.txt {container id}:/app/{target dir}/
```
### 上传数据
执行命令
```
./devel/lib/server/set_data
```
### 更新 服务

注意，需要使用 公司的 k8s config
```
kubectl -n ridesharing delete pod render-map-service-v2-856dc576c-l7l2c
```