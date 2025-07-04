
AVP 自动泊车系统和地图引擎的开发

## 硬件设备

传感器包括 4x鱼眼相机(这部分可能有出入，参考特斯拉的相机数量，前视x3:广角，中焦和长焦，前侧x2， 后侧x2， 后视x1)，IMU，轮速计

处理器 X

## 系统结构

AVP 主要流程

环绕的鱼眼相机 ----> 生成 IPM 图像 ----> 特征检测 ----> 本地建图 ----> 闭环检测 ----> 全局建图

IMU + 轮速记 ----> 里程计

里程计 + 特征检测 + 全局建图 ----> 定位 ----> 扩展卡尔曼滤波 ----> 输出6-DoF 位姿

### 视觉构图

鱼眼相机图片 ----> 去畸变 ----> BEV鸟瞰图拼接 ----> 语义分割

语义元素：车位线，车位角点，减速带，车道线，地面标识，建筑物，轮档，周围车辆

局部语义信息 ----> 投影到车体坐标系 ----> 多帧累积 ----> 局部地图

回环检测优化 ----> 全局一致性优化 ----> 优化后的位姿 ----> 全局地图 ----> 保存地图数据

​​实时感知​​ ----> ​​生成局部描述子​​ ----> ​​全局搜索匹配​​ ----> ​​候选区域筛选​​ ----> ​​几何验证​​ ----> ​​输出精确位姿​

## 工程流程

基于VINS-Fusion的基础上，

多个相机首先会去畸变并进行时间同步，然后通过 IPM（逆透视变换）BEV鸟瞰图拼接，生成BEV图像识别车位线，车位角点，减速带，车道线，地面标识，建筑物，轮档，周围车辆等语义元素。
这部分我不了解

定位这部分会根据光流法确定相机的初始位姿，并结合IMU预积分和路速计速度误差进行因子图优化，得到6-DoF位姿。

将得到的位姿加入到位姿图中，并根据当前的位姿对语义元素进行局部建图，生成局部地图。

检测当前的描述子是否在全局描述子中存在，如果存在，则进行回环检测优化，如果不存在，则在地图引擎中进行查找匹配，

### 特征提取

特征提取的流程：