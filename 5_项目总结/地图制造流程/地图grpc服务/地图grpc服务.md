## code


[https://gitlab.allride-ai.cn/map/map_service/-/tree/grpc](https://gitlab.allride-ai.cn/map/map_service/-/tree/grpc)



## Road Map


Road Map Service中包含了GetLane、GetLanes、GetNearstLanes、GetHeading、GetRoutePoint、GetRoutePointVersion、Route、MapUpdate、GetSignals接口



其中GetLane、GetLanes、GetNearstLanes、GetHeading、GetSignals、Route接口功能和common中的功能一致，具体可见：

[https://gitlab.allride-ai.cn/common/common/-/blob/master/src/map/hdmap/road_map.h](https://gitlab.allride-ai.cn/common/common/-/blob/master/src/map/hdmap/road_map.h)



### MapUpdate


#### 地图增量更新服务


目前车端的地图数据是通过增量更新的方式部署的。

车端product启动后，会启动一个postgres数据库容器，同时该容器会挂载一个外部卷，卷中存放地图数据。



#### 数据增量更新流程图
![](https://cdn.nlark.com/yuque/0/2022/png/22618291/1649993065225-fe58148e-13b1-4d0f-87c8-4664e1f5c349.png)

每次车端更新数据时，会获取本地数据库中所有的tile id以及对应的md5值，将其与目标数据版本的版本号发送到云端，云端从缓存或者数据库中查找相应的数据，将有变化的tile数据发送会车端client，client再对本地数据库完成更新。



### GetRoutePointVersion & GetRoutePoint


这两个接口用于常态化路侧 -随机点获取功能



![](https://cdn.nlark.com/yuque/0/2022/png/22618291/1649993616794-8c32d1b2-789e-43b7-826e-6c22ae7d25f1.png)

服务结构入上图所示。



其中随机点数据库在阿里云数据库中，登录方式：

```plain
psql -p "1433" -d "random_point_map" -h "pgm-uf636020mg487211117910.pg.rds.aliyuncs.com" -U "mapadmin"
```



可以通过set data脚本来更新数据库数据，但由于目前随机点数据没有版本管理，所以每次更新都会将之前的数据覆盖，并且grpc服务中的本地缓存没有效验功能，所以每次更新数据库需要重启grpc服务。

## Render Map


Render Map Service中包括了GetInfo和GetTiles两个接口



### GetInfo
```plain
message GetInfoResponse {
  ResponseHeader header = 1;

  message MapRect {
    Vector2i top_left = 1;
    Vector2i bottom_right = 2;
  }

  MapRect map_rect = 2;
  uint32 render_tile_size = 3;
  uint32 max_tile_level = 4;
  uint64 max_tile_size = 5;
  bytes version = 6;
}
```

包含了render map的范围，包含tile的数量，最高层级，版本号等内容。



### GetTiles
```plain
message GetTileRequest {
  RequestHeader header = 1;
  repeated GeoTileId tile_ids = 2;
}

message GetTileResponse {
  ResponseHeader header = 1;
  repeated render_map.RenderMapTile tiles = 2;
}
```

查询过程，传入tile_ids，传出相关的数据。



### 框架
![](https://cdn.nlark.com/yuque/0/2022/png/22618291/1649928258531-508e9c12-9eb5-47ae-bd0a-de37b70a17e6.png)

在rpc和数据库之间，还有一个redis层来充当缓存，减小数据库压力，redis通过lru策略来动态更新缓存数据。

## 文件


### 配置文件：


目前所有service都使用一套代码，通过修改配置文件来配置是否启动服务，以GetSignals服务为例，GetSignals的配置文件如下：

```plain
url {
  host: "0.0.0.0"
  port: "50059"
} # grpc服务端口

road_map_service {
  is_register: true #是否注册road map服务，如果是false，不启动任何road map service
  config_file: "/cfg/hdmap.cfg" #本地数据库地址
  update_config_file: "/app/cfg/hdmap_cloud.cfg" #增量更新服务云端数据库地址
  rpcs {
    has_get_lane: false #是否启动该服务，false不启动
    has_get_route_point: false
    has_route: false
    has_map_update: false
    has_get_nearest_lanes: false
    has_get_lanes: false
    has_get_heading: false
    has_get_signals: true
  }

  regions {
    name: "suzhou"
    config_file: "/cfg/hdmap_suzhou.cfg" #get_heading服务使用，用于确认是那个地区的数据
  }
  regions {
    name: "beijing"
    config_file: "/cfg/hdmap_beijing.cfg"
  }
  regions {
    name: "jiangning"
    config_file: "/cfg/hdmap_jiangning.cfg"
  }

  route_point_files {
    name: "suzhou"
    config_file: "/app/cfg/route_point.cfg" #get_route_point服务使用，云端数据库位置
  }
}

render_map_service {
  is_register: false #是否注册render map服务，如果是false，不启动任何render map service
  db_config_file: "/cfg/render_map_db.cfg" #数据库地址
  cache_config_file: "/cfg/render_map_cache.cfg" #缓存地址
  rpcs {
    has_get_tiles: false
    has_get_info: false
  }
}

thread_pool_num: 100

```

为了灵活修改配置文件，目前将配置文件都放在oss://allride-sharing/ZC/目录下，docker启动时回去拉取对应目录下的配置文件来选择如何启动服务



oss://allride-sharing/ZC/grpc_cfg_route_v2/         Route service

oss://allride-sharing/ZC/grpc_map_get_siganls/  GetSiganls

oss://allride-sharing/ZC/grpc_map_update_cfg/   MapUpdate

oss://allride-sharing/ZC/grpc_map_update_cfg_v2/   render map service（之前命名有问题，后续可以修改一下）



### 部署文件


[https://gitlab.allride-ai.cn/map/map_deploy/-/tree/v2/service](https://gitlab.allride-ai.cn/map/map_deploy/-/tree/v2/service)

[https://gitlab.allride-ai.cn/map/map_service/-/tree/grpc/deploy](https://gitlab.allride-ai.cn/map/map_service/-/tree/grpc/deploy)



road map：

[https://gitlab.allride-ai.cn/map/map_service/-/blob/grpc/deploy/grpc_loadbalance.yaml](https://gitlab.allride-ai.cn/map/map_service/-/blob/grpc/deploy/grpc_loadbalance.yaml)



render map：

[https://gitlab.allride-ai.cn/map/map_deploy/-/blob/v2/service/k8s_render_map_service/service-render-map-cache.yaml](https://gitlab.allride-ai.cn/map/map_deploy/-/blob/v2/service/k8s_render_map_service/service-render-map-cache.yaml)

