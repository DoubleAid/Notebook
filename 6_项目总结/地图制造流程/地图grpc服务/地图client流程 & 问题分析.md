### 地图数据库
![](https://cdn.nlark.com/yuque/0/2022/png/22618291/1642247501298-7857fd17-15aa-414b-ac78-57307e8b0514.png)

目前地图数据库中包含三个表：

1、road_map，地图基本元素信息；

2、road_map_smoothed，refline信息；

3、environment_map, roi数据。

其他节点通过地图配置文件hdmap.cfg初始化地图client调用地图数据。

### 地图Client
#### client类图 & 简图
![](https://cdn.nlark.com/yuque/0/2021/jpeg/1503704/1640316432023-498e7273-00ed-4d69-9eea-823ae95c48d6.jpeg)

![](https://cdn.nlark.com/yuque/0/2022/png/22618291/1642246803848-28d7a12c-c54d-46c4-945b-5464d8b5bdb6.png)

Map Client对外提供三个接口：

1、getMetadata： 获取数据库表中的所有key值，即地图的所有tile id

2、getTile： 输入tile id，获取tile data。client会先从LRU Cache中查找是否存在该数据，如果不存在，再去PG中查找数据，并将该数据插入LRU Cache。

2、updatePosition： 输入一个物理位置，更新LRU Cache， 以该点所在tile为中心，n个tile为半径，将这部分tile数据加入LRU Cache。该步骤为异步加载。

#### 配置文件
```shell
client {
  db {
    host: "localhost"
    port: 5432
    db: "map"
    table: "environment_map"
  }
  cache {
    radius: 4
    size: 64
  }
}
```

1. db中主要是数据库信息
2. cache中radius为每次调用updatePostion时，需要加载到LRU Cache的tile的数据范围；size为LRU Cache的最大容量

### 
#### 说明
1、每个client对应数据库中的一张表，如果某个节点需要访问两张表，则需要维护两个client。

2、每次updatePosition时，会将该点存入缓存，下一次调用updatePosition时，如果距离小于一定值，则直接返回，不进入流程。目前默认值为100m。

3、为了保证数据的线程安全，在getTile中使用双重检查锁。



### 数据解压&反序列化
#### 说明
数据库中储存二进制数据，需要经过若干个步骤后才能转化为实际使用的tile data。

#### 流程简图
![](https://cdn.nlark.com/yuque/0/2022/png/22618291/1642146888369-0bdd262b-0157-4dd5-8840-5685795850e6.png)

#### ENV Map
输入： 二进制数据

输出： c++结构体 EnvMapTile



proto数据

```plain
message ROI {
  int32 id = 1;
  repeated int32 z = 2;  // unit : cm
  repeated int32 edges = 3; // unit : grid
}

repeated ROI roi = 18;
```

c++ 

```plain
struct RoiProperty {
  RoiProperty() = default;
  RoiProperty(double z, int32_t edge)
      : z(z), edge(edge) {}
  double z;
  int32_t edge;
};

struct EnvMapTile {
  std::vector<std::vector<RoiProperty>> roi_map; 
};
```

#### 说明
1、env map将每个tile分成1000 * 1000个cell，每个cell中填充对应的roi数据。

2、proto为压缩数据，将cell的坐标简化为int值。

3、为提高搜索效率，EnvMapTile中使用数组来储存roi数据。



### 目前问题
#### 表现
感知在使用getTile时出现卡顿。

#### 原因
感知需要使用getTile调用某个tile的数据，正常情况该数据在之前的updatePoistion时已经加载完成，但现在发现LRU Cache中还没有这个tile的数据，后续发现是上次的updatePoistion还没有完成导致的。

由于getTile中使用双重检查锁，会等待updatePoistion结束后再继续调用，导致出现卡顿。



### 解决方案
#### 1、修改cache中radius和size
增加radius，修改成5个tile。

增加载入半径后，可以提前100m将数据加入到LRU Cache中，给updatePoistion留下充分的时间。

周五进行过测试，跑了两圈没有出现之前的问题。

优点：只需要修改配置文件

缺点：增加了内存消耗

[https://allride.yuque.com/xgdyp5/gzgih4/pfvf5b](https://allride.yuque.com/xgdyp5/gzgih4/pfvf5b) （地图client内存测试）



#### 2、缩小updatePosition的默认触发距离
将默认距离修改为20米，目的也是为了给updatePoistion留下充分的时间。



#### 3、修改updatePoistion时周边tile的判定方式
当前方案是以该点所在tile为中心，n个tile为半径加载数据。

如果该点是车在即将驶出所在tile边缘的位置，会导致LRU Cache更新出现滞后，可以将方案改成该点附近350米内的所有tile来避免这种情况。



目前已经将方案2和方案3组合出了一个版本，等待后续测试结果。



#### 4、删除bzip2步骤
bzip2解压占用计算资源，去掉后可以加快数据加载速度

下图是包含解压和反序列化的具体时间消耗：

![](https://cdn.nlark.com/yuque/0/2021/png/22618291/1638156767096-a4d96144-005e-464d-b288-be65505b9d99.png)

#### 5、后续讨论分析




