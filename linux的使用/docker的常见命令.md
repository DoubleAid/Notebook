
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