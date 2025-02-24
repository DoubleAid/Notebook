**1、什么是bev空间**

**<font style="color:rgb(163, 163, 163);">BEV是鸟瞰图（Bird’s Eye View）</font>**<font style="color:rgb(163, 163, 163);">的简称，也被称为上帝视角，是‍‍一种用于描述感知世界的视角或坐标系（3D），BEV也用于代指在计算机视觉领域内的一种‍‍端到端的、由神经网络将‍‍视觉信息，从图像空间转换到BEV空间的技术。</font>

<font style="color:rgb(163, 163, 163);">在高等级智能驾驶领域，除了特斯拉和mobileye走的是纯视觉技术路线外，其他大多数玩家走的还是多传感器融合的技术路线。目前的传感器融合方法有后融合（目标级融合）、前融合（数据级融合）和中融合（特征级融合）。</font>

<font style="color:rgb(163, 163, 163);">相比于后融合和前融合，在BEV空间内进行中融合具有如下优势：</font>

**<font style="color:rgb(134, 134, 134);">1)跨摄像头融合和多模融合更易实现</font>**

<font style="color:rgb(163, 163, 163);">传统跨摄像头融合或者多模融合时，因数据空间不同，需要用很多后处理规则去关联不同传感器的感知结果，操作非常复杂。在BEV空间内做融合后，再做目标检测，算法实现更加简单，BEV空间内视觉感知到的物体大小和朝向也都能直接得到表达。</font>

**2)时序融合更易实现**

<font style="color:rgb(163, 163, 163);">在BEV空间时，可以很容易地融合时序信息，形成4D空间。</font>

<font style="color:rgb(163, 163, 163);">在4D空间内，感知网络可以更好地实现一些感知任务，如测速等，甚至可以直接输出运动预测（motion prediction）给到下游的决策和规控。</font>

**3)可“脑补”出被遮挡区域的目标**

<font style="color:rgb(163, 163, 163);">因为视觉的透视效应，2D图像很容易有遮挡，因而，传统的2D感知任务只能感知看得见的目标，对于遮挡完全无能为力，而在BEV空间内，可以基于先验知识，对被遮挡的区域进行预测，从而“脑补”出被遮挡区域可能存在物体。虽然“脑补”出的物体，有一定“想象”的成分，但这对于下游的规控模块仍有很多好处。</font>

**4)更方便端到端做优化**

**2、如何实现从perspective view 到bev空间的转换**

传统方法<font style="color:rgb(163, 163, 163);">一般是先在图像空间对图像进行特征提取，生成分割结果，然后通过IPM（</font>**<font style="color:rgb(163, 163, 163);">Inverse Perspective Mapping</font>**<font style="color:rgb(163, 163, 163);">，</font>**<font style="color:rgb(163, 163, 163);">逆透视变换</font>**<font style="color:rgb(163, 163, 163);">）函数转换到BEV空间。</font>

**<font style="color:rgb(163, 163, 163);">2.1IMP</font>**

透视变换，将一个平面通过一个投影矩阵投影到指定平面上。

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535492443-10876da1-1528-4e88-9af7-8b029e1474da.png)

输入：至少四个对应点对，不能有三点及以上共线，不需要知道摄相机参数或者平面位置的任何信息。

数学原理：利用点对，求解透视变换矩阵，其中map_matrix是一个3×3矩阵，所以可以构建一个线性方程组进行求解。如果大于4个点，可采用ransac的方法进行求解，一般具有更好的稳定性。

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535504931-705f5d28-6c70-4662-83d6-ae50d1e11094.png)

选点方法：一般采取手动选取，或者利用消影点（图像上平行线的交点，也叫消失点，vanish point）选取。opencv实现。

计算变换矩阵：

<font style="color:rgb(0, 0, 0);background-color:rgb(250, 250, 250);">M = cv2.getPerspectiveTransform(src, dst)</font>

<font style="color:rgb(0, 0, 0);background-color:rgb(250, 250, 250);">Minv = cv2.getPerspectiveTransform(dst, src)</font>

获取IPM图像：<font style="color:rgb(0, 0, 0);background-color:rgb(250, 250, 250);">warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)</font>

<font style="color:rgb(0, 0, 0);background-color:rgb(250, 250, 250);"></font>

**2.2简化相机模型的逆透视变换**

Paper：Adaptive Inverse Perspective Mapping for Lane Map Generation with SLAM

github：[https://github.com/visionbike/AdaptiveIPM](https://github.com/visionbike/AdaptiveIPM)

博客：[https://blog.csdn.net/u013019296/article/details/120170620](https://blog.csdn.net/u013019296/article/details/120170620)

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535516770-3cf0b425-31b9-43df-bd7e-648749b97924.png)

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535121721-48cc8853-2ded-4763-8453-650d11cf9550.png)

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535543809-84bfae40-6b18-42a2-990d-008b646c3b2d.png)

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535557415-aeed8ea7-93ec-4fbf-81de-1c0baf46354b.png)

<font style="color:rgb(0, 0, 0);">邻帧中加入了俯仰角的修正，</font>考虑帧间自适应变换：

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535590620-9e675269-b5d1-469b-910c-39d37d432cfa.png)



![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535600505-eea62365-9017-41a8-8112-e8b60f718766.png)

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535607065-0f14ee47-10c8-47df-a261-e0275810a9be.png)

该方法采用了自适应的IPM模型<font style="color:rgb(77, 77, 77);">，利用运动信息将原始的图像信息精准地转换为鸟瞰图，利用了单目的视觉SLAM的方法得到的运动信息。该方法即使在行驶过程中有较大的运动也可以提供稳定的鸟瞰图。</font>

**基于深度学习的方法：**

**2.3、Lift-Splat系列**

<font style="color:rgb(118, 118, 118);">经典的2d转3d的方法：Lift-Splat-Shoot（LSS）</font>

<font style="color:rgb(118, 118, 118);">将二维图像特征生成3D特征（这一步对应论文中的Lift操作），然后把3D特征“拍扁”得到BEV特征图（这一步对应论文中的Splat），最终在BEV特征图上进行相关任务操作（这一步对应于Shooting）。具体来说，二维图像特征生成3D特征这一步使用了“视锥”点云的操作，如下图所示。</font>

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535622998-bb263e41-ba9f-4337-8a99-097225a5ef0b.png)

<font style="color:rgb(118, 118, 118);">准确的深度估计是非常困难的，LSS将图像回归变成了分类，提前设定好一系列的深度锚点。优点：每个像素都能够获得一个深度估计值，缺陷就是深度的分布是离散的。后续基于此方法的研究有BEVDet、BEVFusion、BEVDepth。</font>

**2.4、MLP系列（**<font style="color:rgb(118, 118, 118);">Multi-Layer Perceptron</font>**）**

Cross-View Semantic Segmentation for Sensing Surroundings （vpn）<font style="color:rgb(118, 118, 118);">使用全连接网络来让network自己学习如何进行视角的变换。</font>

<font style="color:rgb(118, 118, 118);">MLP优点比较明显，实现非常简单。但是缺点是相机的内外参是重要的先验信息，MLP放弃了这些信息，采用数据驱动的方式隐式的学习内外参，并将其融入到MLP的权重当中。</font>

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535637757-83170c15-7aaf-44b3-9e69-6798fb774a14.png)

           多层感知机概念图

HDMapNet借鉴了VPN的方法。不同点在在于后者考虑了相机的内外参数和自车位姿态。**通过以当前位置为（0,0）点，构建一个（-15,-30，15,30）的区域，并假设高程z为0。然后在x、y方向以0.15米的分辨率插值出200×400的点构建bev的平面。图像经过cnn卷积之后的宽高为40×80。通过相机外参，将bev平面（三维的）分别转到6个图像的平面通过双线性插值出bev的图像特征。**然后对这个六个bev特征进行max计算得到一个完整的bev特征图。

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535649347-b827f11c-b56b-4ecf-b3c0-8f4bf9b961f8.png)

                                                               图像空间映射到BEV空间

**2.5、Transformer系列**

**BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**

**为什么用Transformer？**

自2020年中以来，transformer席卷计算机视觉领域，由于使用了全局注意力机制，transformer更适合执行视图转换工作。目标域中每个位置具有相同的距离来访问原域中的任何位置，克服了CNN中卷积层感受野受限局部。

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535660450-a5b10a95-ba48-49a7-a0fc-b721b8d24508.png)

**<font style="color:rgb(118, 118, 118);">                                                   Transformer架构图</font>**

Positional Encoding （位置编码）

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535674169-ce76004e-1170-49af-b066-bb93250ea17c.png)

Self-attention（自注意力机制）

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535684707-1f89f164-8e57-42bf-8419-2af6d47ac284.png)

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535691612-a7389520-9787-4378-8cc6-315caaa3d7a6.png)

Add & Norm（残差连接/层标准化）

Feed-Forward。

在Transformer中前馈全连接层就是具有两层线性层的全连接网络。<font style="color:rgb(18, 18, 18);">考虑注意力机制可能对复杂过程的拟合程度不够, 通过增加两层网络来增强模型的能力。</font>

**<font style="color:rgb(118, 118, 118);">为什么要用时空融合？</font>**

<font style="color:rgb(118, 118, 118);">时序信息对于自动驾驶感知任务十分重要，时序信息一方面可以作为空间信息的补充，来更好的检测当前时刻被遮挡的物体或者为定位物体的位置提供更多参考信息。除此之外时序信息对于判断物体的运动状态十分关键，先前基于纯视觉的方法在缺少时序信息的条件下几乎无法有效判断物体的运动速度。</font>

**<font style="color:rgb(118, 118, 118);">BEVFormer的核心是：基于多视角和时序BEV特征迭代优化，获得高精度BEV特征，即作为一个Backbone/neck来使用。</font>**

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535706390-456e51ff-9c13-44ca-a200-84b1fb969878.png)

**<font style="color:rgb(118, 118, 118);">                                                            BEVFormer网络架构图</font>**

时序自注意力机制Temporal Self-Attention。将上一帧的bev特征作为当前帧的输入。在特征采样和融合的过程中，只考虑与当前位置靠近的特征进行融合。

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535717678-6a8ed9c4-434d-4e0f-bd0d-3d159f2a24ee.png)

<font style="color:rgb(18, 18, 18);">空间交叉注意力机制Spatial Cross-Attention。在z轴上采样4个高度值，获得3d空间下的采样点，则会覆盖不同的高度例如-3米到3米。让后用相机的内外参将3d点投影到2d点，由于投影点不太准，需要在投影点作为先验在附近进行采样。每个视图只提取跟他交互的bev。</font>

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535728093-fba1ed93-97ce-4920-9f0d-a0ae673a02fe.png)

**3、bev空间下语义地图感知的框架**

<font style="color:rgb(163, 163, 163);">第一步，先将摄像头数据输入到共享的骨干网络（Backbone），提取每个摄像头的数据特征（feature）。</font>

<font style="color:rgb(163, 163, 163);">第二步，把所有的摄像头数据（跨摄）进行融合，并转换到BEV空间。</font>

<font style="color:rgb(163, 163, 163);">第三步，在BEV空间内，进行跨模态融合，将像素级的视觉数据和激光雷达点云进行融合。</font>

<font style="color:rgb(163, 163, 163);">第四步，进行时序融合，形成4D时空维度的感知信息。</font>

<font style="color:rgb(163, 163, 163);">最后一步，就是多任务输出，可以是静态语义地图、动态检测和运动预测等，给到下游规控模块使用。</font>

HDMapNet网络框架为例：

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535745546-52bebf6d-a2b3-4411-8fcb-7aadd8714cbc.png)

 HDMapNet架构图

+ <font style="color:rgb(110, 110, 110);">图像特征提取：efficientNet-B0</font>
+ <font style="color:rgb(110, 110, 110);">激光雷达特征提取：PointPillar</font>
+ <font style="color:rgb(110, 110, 110);">图像特征投影到BEV空间: MLP</font>
+ <font style="color:rgb(110, 110, 110);">激光雷达和图像的特征融合：torch.cat()操作</font>
+ <font style="color:rgb(110, 110, 110);">BEV特征提取和融合：resnet18+上采样</font>
+ <font style="color:rgb(110, 110, 110);">多任务头：语义分割（3）、实例分割（16）、方向预测分类（36）</font>
+ <font style="color:rgb(110, 110, 110);">后处理：cluster</font>

精度评估：

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535754532-9b0d6639-5b50-4210-8d9f-8775b6c6482b.png)

训练结果：

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535764500-2fb18c1f-7951-4bd7-853c-2adc46ef1b0c.png)

                                     模型推理结果                                                                                    高精度地图

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535793564-1ad6be66-5e80-4f7d-90c8-06f6e237d11c.png)

实验结果：

![](https://cdn.nlark.com/yuque/0/2022/png/873758/1660535802914-7abe67bf-a7d0-4db8-9efb-6197635b970b.png)

                         融合结果>相机>激光雷达 （车道线、人行横道线）；融合结果>激光雷达>相机 （路沿）

HDMapNet框架除了第四步时序融合没有其他都是全的，且图像和激光雷达这两个输入是解耦的。

进可以用来做车端的道路结构信息的实时检测， 退可以在网络里面加入时序信息做离线场景的结构化提取。

**4、bev技术的局限与挑战**

**4.1、数据问题**

<font style="color:rgb(163, 163, 163);">BEV感知中最具备挑战的还是如何获取更多维度的数据，以及产生更高质量的真值。加上Transformer本身的特性，为更好地发挥优势，其对数据量的要求也比传统卷积神经网络大得多，这就越发加剧了模型对数据的“饥渴”程度。</font>

**4.2、算力消耗问题**

<font style="color:rgb(163, 163, 163);">由于使用Transfomer进行BEV空间转化非常消耗算力，对车端有限算力提出了挑战。图像处理中，使用Transformer的计算复杂度与图像尺寸的平方成正比，这会导致，在图像很大的时候，计算量过于庞大。</font>

**<font style="color:rgb(163, 163, 163);">4.3、bev算法更复杂</font>**

<font style="color:rgb(163, 163, 163);">相比于传统2D图像检测，BEV感知算法会复杂得多，尤其是前文提到的云端的3D重建、4D空间的标注、真值生成和模型训练，都是之前2D感知任务中所没有的，相应地难度和门槛自然也要高一些。</font>
