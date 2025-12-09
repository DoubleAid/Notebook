# MSCKF 多状态约束卡尔曼滤波器 Multi-State Constraint Kalman Filter

首先明确 MSCKF 的核心应用场景：视觉惯性里程计（VIO）

+ 输入是 IMU 原始数据（角速度 $\omega$、线加速度 a）和相机图像（特征点观测）
+ 输出是相机 / IMU 的实时位姿（$R, t$）、速度（v）和传感器零偏（$b_w, b_a$）

## 一、先铺垫：MSCKF 的状态向量（为什么叫 “多状态”？）

MSCKF 的核心创新是 “增广状态向量”—— 它不把环境特征点放进状态（避免维数爆炸），而是把 “滑动窗口内的多个历史相机位姿” 加进去，形成 “多状态”。这是它看起来像后端优化的关键，先明确状态向量的构成：

### 1. 基础状态（IMU 核心状态，15 维）

IMU 是 VIO 的 “高频数据源”，其状态是 MSCKF 的核心，和传统 EKF-VIO 一致：

$$
\mathbf{x}_{\text{imu}} = \begin{bmatrix} \mathbf{p} & \mathbf{v} & \mathbf{q} & \mathbf{b}_w & \mathbf{b}_a \end{bmatrix}^T
$$

+ $\mathbf{p} \in \mathbb{R}^3 ：IMU 相对于世界坐标系的位置；
+ $\mathbf{v} \in \mathbb{R}^3$：IMU 的线速度；
+ $\mathbf{q} \in \mathbb{R}^4$：IMU 相对于世界坐标系的姿态（单位四元数）；
+ $\mathbf{b}_w \in \mathbb{R}^3$：陀螺仪零偏（缓慢漂移，需估计）；
+ $\mathbf{b}_a \in \mathbb{R}^3$：加速度计零偏（缓慢漂移，需估计）。

### 2. 增广状态（滑动窗口内的历史相机位姿，6K 维）

MSCKF 会维持一个 滑动窗口（比如窗口大小 $K=5$），窗口内包含 “当前帧 + 最近 4 帧” 的相机位姿（相对于 IMU 的外参已知，记为 $T_{ic} = (R_{ic}, t_{ic})$），每个相机位姿用 6 维表示（3 维位置 + 3 维姿态）：

$$\mathbf{x}_{\text{cam}} = \begin{bmatrix} \mathbf{p}_{c1} & \mathbf{q}_{c1} & \mathbf{p}_{c2} & \mathbf{q}_{c2} & ... & \mathbf{p}_{cK} & \mathbf{q}_{cK} \end{bmatrix}^T
$$

+ $\mathbf{p}_{ck} \in \mathbb{R}^3$：第 k 帧相机的位置（世界坐标系）；
+ $\mathbf{q}_{ck} \in \mathbb{R}^4$：第 k 帧相机的姿态（世界坐标系）；
+ 相机位姿与 IMU 状态的关系（外参转换）：
    $$
    \mathbf{p}_{ck} = \mathbf{p} + R \cdot \mathbf{t}_{ic}, \quad \mathbf{q}_{ck} = \mathbf{q} \otimes \mathbf{q}_{ic}
    $$
    （$\otimes$ 是四元数乘法，$R_{ic}$ 是 IMU 到相机的旋转矩阵，由外参标定得到）

### 3. 完整状态向量（15 + 6K 维）

$$
\mathbf{x} = \begin{bmatrix} \mathbf{x}_{\text{imu}}^T & \mathbf{x}_{\text{cam}}^T \end{bmatrix}^T
$$

+ 滑动窗口的作用：固定 K（比如 5），当新帧到来时，移除最旧的相机位姿，加入新帧相机位姿，保证状态维度不随时间增长（解决传统 EKF-SLAM 维数爆炸问题）。

## 二、MSCKF 的核心流程：滤波器的 “预测 - 更新” 本质

MSCKF 完全遵循 EKF 的递推逻辑：先通过 IMU 数据做 “预测步”（外推状态和协方差），再通过视觉特征点观测做 “更新步”（用多状态约束修正状态）。下面结合公式拆解每一步：

### 步骤 1：预测步（IMU 积分，高频递推）

预测步和传统 EKF-VIO 完全一致 —— 利用 IMU 的高频数据（通常 1000Hz），外推当前的 IMU 状态和协方差矩阵，是 “递推式” 计算（实时输出，不回溯）。

#### （1）状态预测（IMU 运动模型）

IMU 的原始测量值会先扣除零偏：

$$
\omega_{\text{true}} = \omega_{\text{meas}} - \mathbf{b}_w, \quad a_{\text{true}} = a_{\text{meas}} - \mathbf{b}_a
$$

然后通过 IMU 积分更新状态（离散时间下的欧拉积分，简化版）：

$$
\begin{cases}
\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \delta \mathbf{q}(\omega_{\text{true}} \cdot \Delta t) \\
\mathbf{v}_{k+1} = \mathbf{v}_k + (R_k \cdot a_{\text{true}} + \mathbf{g}) \cdot \Delta t \\
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \cdot \Delta t + \frac{1}{2}(R_k \cdot a_{\text{true}} + \mathbf{g}) \cdot \Delta t^2 \\
\mathbf{b}_{w,k+1} = \mathbf{b}_{w,k} \quad (\text{零偏缓慢漂移，预测时视为不变}) \\
\mathbf{b}_{a,k+1} = \mathbf{b}_{a,k}
\end{cases}
$$

+ $\delta \mathbf{q}(\cdot)$：由角速度积分得到的微小旋转四元数；
+ $\mathbf{g}$：重力加速度向量；
+ $\Delta t$：IMU 测量时间间隔。

#### （2）协方差预测（不确定性传播）

EKF 的核心是 “用协方差矩阵 $\mathbf{P}$ 描述状态不确定性”，预测步通过雅可比矩阵传播协方差：

$$
\mathbf{P}_{k+1|k} = \mathbf{F}_k \cdot \mathbf{P}_{k|k} \cdot \mathbf{F}_k^T + \mathbf{Q}_k
$$

+ $\mathbf{P}_{k|k}$：第 k 步更新后的协方差矩阵（15+6K 维）；
+ $\mathbf{F}_k$：状态转移雅可比矩阵（描述状态变化对误差的影响）；
+ $\mathbf{Q}_k$：IMU 测量噪声矩阵（已知，由传感器标定得到）。

+ 关键：预测步是纯滤波器逻辑 —— 实时递推，不依赖视觉观测，也不回溯历史状态。

### 步骤 2：更新步（视觉约束，多状态联动修正）

更新步是 MSCKF 看起来像 “后端优化” 的核心 —— 当视觉特征点跟踪丢失时（比如跟踪了 5 帧后消失），用该特征点在滑动窗口内所有相机位姿下的观测，构建 “多状态约束”，修正整个状态向量。

#### （2.1）视觉观测模型：特征点重投影误差

假设一个特征点 $\mathbf{X} \in \mathbb{R}^3$（世界坐标系下），在滑动窗口内的 K 个相机帧中都有观测（像素坐标 $\mathbf{u}_{k,i} = (u_{k,i}, v_{k,i})^T$，$i=1..K$）。

对于第 i 帧相机，特征点 $\mathbf{X}$ 的重投影过程是：

1. 世界坐标系 → 相机坐标系：$\mathbf{X}_{c,i} = R_{c,i}^T (\mathbf{X} - \mathbf{p}_{c,i})$（$R_{c,i}$ 是第 i 帧相机的旋转矩阵）；
2. 相机坐标系 → 像素坐标系（透视投影）：$\mathbf{u}_{k,i} = \begin{bmatrix} \frac{f_x \cdot X_{c,i,x}}{X_{c,i,z}} + c_x \\ \frac{f_y \cdot X_{c,i,y}}{X_{c,i,z}} + c_y \end{bmatrix} + \mathbf{n}_{i}$

+ $(f_x, f_y, c_x, c_y)$：相机内参（标定已知）；
+ $\mathbf{n}_i \in \mathbb{R}^2$：视觉观测噪声（高斯噪声，协方差 $\sigma^2 \mathbf{I}$）。

#### （2.2）多状态约束：特征点三角化 + 约束构建

由于特征点 $\mathbf{X}$ 不在状态向量中，MSCKF 先通过滑动窗口内的相机位姿，对 $\mathbf{X}$ 做三角化（得到 $\hat{\mathbf{X}}$），再用 $\hat{\mathbf{X}}$ 构建 “重投影误差约束”—— 这个约束会关联滑动窗口内的所有相机位姿（属于状态向量的一部分），进而修正 IMU 核心状态。

重投影误差残差定义为：

$$
\mathbf{r}_i = \mathbf{u}_{k,i} - \pi(\mathbf{p}_{c,i}, R_{c,i}, \hat{\mathbf{X}})
$$

+ $\pi(\cdot)$：透视投影函数（即上述重投影过程）；
+ 整个特征点的残差向量是 K 个帧残差的拼接：$\mathbf{r} = [\mathbf{r}_1^T, \mathbf{r}_2^T, ..., \mathbf{r}_K^T]^T \in \mathbb{R}^{2K}$。

#### （2.3）EKF 更新：用残差修正状态

EKF 更新的核心是 “将非线性残差线性化，用卡尔曼增益修正状态”，步骤如下：

1. 残差线性化：$\mathbf{r} \approx \mathbf{H} \delta \mathbf{x} + \mathbf{n}$，其中：

+ $\delta \mathbf{x}$：状态向量的微小误差（$\mathbf{x} = \hat{\mathbf{x}} + \delta \mathbf{x}$）；
+ $\mathbf{H} \in \mathbb{R}^{2K \times (15+6K)}$：观测雅可比矩阵 —— 描述状态误差对残差的影响，这是多状态约束的核心：
  + $\mathbf{H}$ 的每一行对应一个像素观测，每一列对应状态向量的一个维度；
  + 由于残差关联了滑动窗口内的所有相机位姿，$\mathbf{H}$ 会有非零元素对应这些相机位姿的状态，从而实现 “一个特征点约束多个状态”。

2. 卡尔曼增益计算：

$$
\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R})^{-1}
$$

+ $\mathbf{R}$：观测噪声协方差矩阵（对角矩阵，对角元素为 $\sigma^2$）。

3. 状态修正：

$$
\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K} (\mathbf{r} - \mathbf{H} \delta \mathbf{x}_{k|k-1})
$$

（注：$\delta \mathbf{x}_{k|k-1} = 0$，因为 $\hat{\mathbf{x}}_{k|k-1}$ 是预测的最优状态，线性化点在 $\hat{\mathbf{x}}_{k|k-1}$）

4. 协方差修正：

$$
\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{P}_{k|k-1}
$$

关键：更新步是 “单次修正” 而非 “迭代优化”—— 虽然用了多状态约束，但只通过一次卡尔曼增益修正状态，没有像后端优化那样反复迭代最小化残差，这是滤波器和后端优化的核心区别。

步骤 3：滑动窗口管理（维持状态维度）

当新的相机帧到来时，若滑动窗口大小超过 K（比如 $K=5$），则移除最旧的相机位姿 —— 移除时会通过 “状态边缘化”（类似后端优化的边缘化，但更简单），将旧位姿的约束信息融入到剩余状态的协方差矩阵中，保证约束不丢失。

边缘化的核心逻辑：将协方差矩阵 $\mathbf{P}$ 分块（旧位姿状态 $\mathbf{x}_{\text{old}}$ 和剩余状态 $\mathbf{x}_{\text{rest}}$），通过舒尔补（Schur complement）剔除 $\mathbf{x}_{\text{old}}$，得到新的协方差矩阵 $\mathbf{P}_{\text{new}}$。

三、用 “具体例子” 理解流程

假设滑动窗口大小 $K=3$，IMU 频率 1000Hz，相机频率 30Hz：

1. 预测：IMU 每 1ms 递推一次状态（$\mathbf{p}, \mathbf{v}, \mathbf{q}, \mathbf{b}_w, \mathbf{b}_a$）和协方差 $\mathbf{P}$；
2. 视觉跟踪：相机每 33ms 输出一帧图像，提取特征点并跟踪，将当前相机位姿加入滑动窗口（窗口内有帧 1、帧 2、帧 3）；
3. 特征点丢失：帧 3 后，某个特征点不再出现（跟踪丢失），此时用该特征点在帧 1、帧 2、帧 3 的像素观测，三角化得到 $\hat{\mathbf{X}}$；
4. 多状态约束更新：构建 3 帧的重投影误差残差（6 维），计算观测雅可比 $\mathbf{H}$（6 行 × (15+18)=33 列），通过 EKF 更新修正整个状态向量（包括 IMU 核心状态和 3 帧相机位姿）；
5. 窗口滑动：新帧 4 到来，移除最旧的帧 1 相机位姿，通过边缘化更新协方差 $\mathbf{P}$，窗口变为帧 2、帧 3、帧 4，重复上述流程。
