# 高精地图数据格式
### OpenDrive
使用XML格式文件来描述道路结构，自动驾驶系统可通过读取XML文件构造路网，座舱域可通过进一步渲染后通过显示屏展示给用户。

OpenDRIVE通过道路参考线（Reference Line）,车道（Lanes）、车道段（Section）、物体（Objects）、交通标志（Road Signals）、
标高（Elevation）、交叉口（Junction）等元素来描述道路结构。

### NDS
NDS采用了数据库技术存储地图数据，在兼顾性能和功能的基础上，可以比较好地解决地图增量更新、数据安全、数据可靠的问题。

一份地图数据可以称为一个NDS数据库（NDS Database），这个数据库是按照NDS标准设计的，只要按照这个标准去制作的地图数据，
都是可以兼容的。一个数据库包含不同的产品数据库（Product Database），这些产品数据库可以是不同图商制作的地图，并且可以
进行独立的版本控制和版本更新。而每一个产品数据库还可以被进一步划分成多个更新区域(Update Region)。

+ 产品数据库：NDS中格式文档本身，格式文档就有上百页，对于地图考虑得非常全面。在NDS地图中有POI数据，它指的是地图上的用户可能感兴趣的一个点，比如一个商铺，一个公园。 它提供了很多描述功能，包括表述语音、基本地图显示，功能非常全面，但是比较复杂。
+ 支持局部更新：NDS支持局部更新，可以对某个特定范围区域（如国家、省、市）进行更新。而且数据库可以分为多个产品，每个产品独立维护更新，最典型的应用就是NDS数据里面可以既包含A公司的基本导航技术数据，又包含B公司的POI（兴趣点）数据，非常适合各优势领域的公司进行合作。
+ Level划分：Level就是我们手机地图里可以看到的比例尺的概念。不同比例尺下，我们可以看到不同颗粒度的地图信息。
+ 地图块：分块是地图领域的一个通用技术。因为地图范围非常大，把整个地图切分为众多小格子，便于地图数据的更新维护。

