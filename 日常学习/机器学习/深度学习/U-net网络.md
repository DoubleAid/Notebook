# U-Net: Convolutional Networks

[参考链接](https://zhuanlan.zhihu.com/p/43927696)
Tag: Biomedical Image Segmentation(生物医学图像分割), Semantic Segmentation(语义分割)

- [U-Net: Convolutional Networks](#u-net-convolutional-networks)
  - [前言](#前言)
  - [1.算法详解](#1算法详解)
    - [1.1 U-Net的网络结构](#11-u-net的网络结构)
    - [1.2 U-Net究竟输入了什么](#12-u-net究竟输入了什么)
    - [1.3 U-Net的损失函数](#13-u-net的损失函数)
  - [2. 数据扩充](#2-数据扩充)
  - [3. 总结](#3-总结)

## 前言

U-Net是比较早的使用全卷积网络进行语义分割的算法之一， 论文中使用包含压缩路径和扩展路径的对称U形结构在当时非常具有创新性， 且一定程度上影响了后面若干分割网络的设计， 该网络的名字也是取自其U形形状

U-Net的实验是一个比较简单的ISBI cell tracking数据集， 由于本身的任务比较简单， U-Net仅仅通过30张图片并辅以数据扩充策略便达到了非常低的错误率

[论文链接](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

## 1.算法详解

### 1.1 U-Net的网络结构
![](image/U-Net网络/u-net-architecture.png)
网络是一个经典的全卷积神经网络（即网络中没有全连接操作）。 网络的输入是一张572x572的边缘经过镜像操作的图片
### 1.2 U-Net究竟输入了什么 
### 1.3 U-Net的损失函数

## 2. 数据扩充

## 3. 总结