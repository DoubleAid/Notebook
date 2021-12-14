# 过程
### 下载相应的 base_map.pb.txt
下载地址： oss://allride-map/data/basemap/suzhou/0.0.122/

### 下载 v5/map-pipeline 分支
v5/map-pipeline 不支持 ubuntu 18 melodic 版本
现使用 master 分支

### 第一步
读取 cfg 文件，生成 HDMap_Processor_Configure 配置类 执行 过程 503

生成的 config 类，包括 
```
type()
input().file().path()
output().file().path()
extra_cfg_path()
```

在 "tile_processor.cpp" 中，根据 config 中的type() 为 ConfigHDMap_Processor_Type_BASE_MAP_LOADER
加载 BaseMapLoader

BaseMapLoader的定义如下：
```cpp
class BaseMapLoader : public TileProcessor {
 public:
  BaseMapLoader(const proto::config::ConfigHDMap_Processor& config);
  ~BaseMapLoader() = default;
  void process() override;
 protected:
  proto::base_map::BaseMap base_map_;
  std::unique_ptr<common::storage::Storage<GeoTileId, proto::map::HDMapTile>> storage_out_;
  common::math::Polygon base_map_scope_;
};
```
BaseMapLoader 构造函数 
+ 判断输入的类型，将 base_map.pb.txt 读取给 base_map_
+ 判断输出的类型，将 输出的文件夹目录给 storage_out_
+ 判断 extra_cfg_path 是否存在，读取 cfg 文件给 base_map_scope

[basemap的proto原型](#basemap的proto原型)

BaseMapLoader 的 process() 过程 (base_map_loader.cpp:181)
+ 给道路的两边添加 虚拟 boundary
+ 给 Crossroad、Junction和roi 绑定规则，转化为三维的模型
+ 绑定 id 和 每个对象的 polygen
+ 对 每一种对象进行分块，将其保存在相应的块中
+ 调用 map/client/tile_file_storage.h 的 set 方法进行保存

保存 成一个meta文件（感觉没啥用，里面是 GeoTileId 的 三点坐标，也就是各个 bin 文件的文件名）
若干个 bin文件，每一个文件都是一个 tile

### 第二步 road map geometry resampled 路线图几何重新采样
新建了 /opt/allride/data/map/road_map_geometry_resampled 文件夹
读取配置文件 map/conf/processor/data_pipeline/geometry_resample.cfg
文件格式
```cfg
type: ROAD_MAP_GEOMETRY_RESAMPLE

input {
  file {
    path: "/opt/allride/data/map/road_map_raw"
  }
  cache {
    radius: 9
    size: 10000
  }
}

output {
  file {
    path: "/opt/allride/data/map/road_map_geometry_resampled"
  }
}

use_multi_thread: true
multi_thread_num: 4
```
同样是包含了 input 和 output
配置的type为 ConfigHDMap_Processor_Type_ROAD_MAP_GEOMETRY_RESAMPLE
创建 GeometryResampleProcessor
继承关系
GeometryResampleProcessor <-- RoadMapProcessor <-- TileProcessor

GeometryResampleProcessor 对 processTile 进行了继承， 其返回值为 RoadMapTilePtr 为 RoadMapTile的指针


```cpp
class RoadMapProcessor {
    // RoadMap 指针
    std::unique_ptr<hdmap_v2::RoadMap> road_map_;
    // 保存
    std::unique_ptr<common::storage::ThreadSafeStorage<GeoTileId, T>> storage_;
    std::ofstream error_file_;
    int32_t kThreadPoolSize = 0;
    // 线程池
    std::unique_ptr<common::util::ThreadPool> thread_pool_;
}
```
执行 road_map_processor 的 process 过程，对每一块都进行 processorJob处理

processorJob中读取了相应的二进制文件，做了 processTile 处理，processTile就是对每一种类别进行resample处理





最后将处理结果进行了保存








# 附加内容
### basemap的proto原型
```proto
message BaseMap {
  map<int64, Lane> lanes = 1;
  map<int64, Boundary> boundaries = 2;
  map<int64, Curb> curbs = 3;
  map<int64, Signal> signals = 4;
  map<int64, WaitingArea> waiting_areas = 5;
  map<int64, StopLine> stop_lines = 6;
  map<int64, ClearZone> clear_zones = 7;
  map<int64, Crosswalk> crosswalks = 8;
  map<int64, GreenBelt> green_belts = 9;
  map<int64, Bumper> bumpers = 10;
  map<int64, WeightSign> weight_signs = 11;
  map<int64, SpeedSign> speed_signs = 12;
  map<int64, HeightSign> height_signs = 13;
  map<int64, Junction> junctions = 14;
  map<int64, Tunnel> tunnels = 15;
  map<int64, Bridge> bridges = 16;
  map<int64, ROI> rois = 17;
}
```
是多个道路元素字典的集合， 其实现的base_map.pb.h 中有其具体的实现(base_map.pb.h:3550)
+ 获取某个元素的字典方法：mutable_xxx(), 例如 mutable_lanes(), mutable_boundaries()
+ 



### HDMapTile的proto原型
```java
message HDMapTile {
  uint64 id = 1; // tile id
  uint64 version = 2;
  uint64 id_generator = 3; // item id generator

  // Lane info
  repeated Lane lanes = 4;
  repeated LaneBoundary lane_boundaries = 5;

  // Geometry shapes
  repeated Line lines = 6;
  repeated Area areas = 7;

  // Rules
  repeated SignalRule signal_rules = 8;
  repeated StopRule stop_rules = 9;
  repeated YieldRule yield_rules = 10;
  repeated SpeedRule speed_rules = 11;
  repeated BumperRule bumper_rules = 12;
  repeated HeightRule height_rules = 13;
  repeated ObstacleRule obstacle_rules = 14;
  repeated CrosswalkRule crosswalk_rules = 15;
  repeated ClearZoneRule clear_zone_rules = 16;
  repeated JunctionRule junction_rules = 17;

  // ROI
  repeated ROI roi = 18;
  bytes compressed_roi = 19;

  // Parking
  repeated ParkingSpot parking_spots = 20;

  // GeoTileId 是一个三点坐标，x,y 表示坐标位置， z 表示 level
  GeoTileId tile_id = 21;

  repeated DigitRule digit_rules = 22;

  repeated Curb curbs = 23;
  repeated WaitingArea waiting_areas = 24;
  repeated ROIRegion roi_regions = 25;

  math.Points ground_samples = 26;

  math.Vector3i origin = 27; //only for storage
  repeated GeoTileId neighbor_tiles = 28; //only for storage
  bool is_conversion_required = 29;//only for storage
}
```

