### 下载相应的 base_map.pb.txt
下载地址： oss://allride-map/data/basemap/suzhou/0.0.122/

### 下载 v5/map-pipeline 分支

v5/map-pipeline 不支持 ubuntu 18 melodic 版本



### 第一步
读取 cfg 文件，生成 HDMap_Processor_Configure 配置类

生成的 config 类，包括 
```
type()
input().file().path()
output().file().path()
extra_cfg_path()
```
调用配置 ConfigHDMap_Processor_Type_BASE_MAP_LOADER
加载 BaseMapLoader

