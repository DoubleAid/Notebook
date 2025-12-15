# IMU预积分的原因和优化推导

IMU 预积分：设计初衷是解决传统绝对积分的工程痛点，最终为后端优化提供精准、可高效更新的相对运动约束。 “核心原因→数学建模→积分推导→离散实现→雅可比与优化” 逐步展开，兼顾理论严谨性与工程落地逻辑。

## 一、为什么需要 IMU 预积分？（核心原因）

IMU 预积分的核心原因是解决**多传感器时间异步、简化优化计算、提升融合精度**，具体可概括为 3 点：

1. 对齐多传感器时间戳：IMU 采样率（100-2000Hz）远高于相机 / 激光雷达（10-30Hz），预积分可将相邻关键帧（如相机帧）之间的 IMU 数据积分成一个 “等效变换增量”（位置、速度、姿态），避免逐帧处理高频 IMU 数据，实现与低帧率传感器的时间同步。
2. 简化非线性优化：SLAM/VIO 的后端优化（如 BA）需频繁调整关键帧位姿，若每次调整都重新积分 IMU 数据，计算量极大。预积分将 IMU 数据与关键帧位姿解耦，仅需一次积分得到增量，后续优化时通过预积分的雅可比矩阵快速更新误差，大幅提升优化效率。
3. 保留 IMU 误差信息：预积分过程中会同步计算积分结果的协方差矩阵和雅可比矩阵，量化 IMU 噪声（零偏、高斯噪声）对积分结果的影响，为传感器融合（如 ESKF、BA）提供可靠的误差估计，提升运动估计的精度和鲁棒性。

### 1. 传统绝对积分的痛点：偏置优化导致 “连锁更新” 灾难

IMU 读数存在零偏（陀螺偏置$\mathbf{b}_\omega$、加计偏置$\mathbf{b}_a$），且偏置是缓慢时变的。传统绝对积分的状态方程为：

$$
\begin{cases}
\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \int_k^{k+1} \frac{1}{2} \mathbf{R}_b^w \otimes (\tilde{\boldsymbol{\omega}}(t) - \mathbf{b}_\omega)^\wedge dt \\
\mathbf{v}_{k+1} = \mathbf{v}_k + \int_k^{k+1} (\mathbf{R}_b^w(\tilde{\mathbf{a}}^b(t) - \mathbf{b}_a) - \mathbf{g}) dt \\
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2} \int_k^{k+1} (\mathbf{R}_b^w(\tilde{\mathbf{a}}^b(t) - \mathbf{b}_a) - \mathbf{g}) dt^2
\end{cases}
$$

可见，所有后续帧的绝对状态都依赖初始偏置估计。若后端优化修正了某一帧的偏置，则该帧之后的所有绝对状态都需要重新积分计算 —— 海量 IMU 数据重新积分，实时性完全无法保证。

传统积分的结果是 “绝对状态”（世界系下的位姿、速度），依赖初始绝对位姿$\mathbf{q}_0, \mathbf{p}_0$。若后端优化调整了初始绝对位姿（如闭环检测修正全局位姿），则所有积分结果都需重新计算，无法直接复用。

### 2. 预积分的核心解决思路

放弃 “计算绝对状态”，转而计算相邻帧间的相对运动增量（$\Delta \mathbf{q}, \Delta \mathbf{v}, \Delta \mathbf{p}$），且增量的计算满足两个关键特性：

1. **不依赖绝对状态**：仅依赖 IMU 读数和偏置，与世界系下的绝对位姿、绝对速度无关；
2. **偏置可快速更新**：预积分是偏置的函数，当偏置优化后，无需重新积分 IMU 数据，仅通过泰勒展开 + 雅可比矩阵即可快速修正增量。

最终，预积分输出 “相对运动约束”，后端优化时直接复用该约束，大幅降低计算复杂度，同时保证精度。

## 二、数学建模与符号约定（推导基础）

### 1. 坐标系定义

+ 世界系（w 系）：惯性系，重力向量$\mathbf{g}^w = [0, 0, g]^T$（$g=9.81m/s²$，向下为正）；
+ IMU 本体系（b 系）：与 IMU 固连，x/y/z 对应 IMU 敏感轴。

### 2. IMU 观测模型（去噪去偏）

IMU 原始读数含噪声和偏置，真实角速度 / 加速度为：

$$
\begin{cases}
\boldsymbol{\omega}(t) = \tilde{\boldsymbol{\omega}}(t) - \mathbf{b}_{\omega}(t) - \boldsymbol{\eta}_{\omega}(t) \\
\mathbf{a}(t) = \tilde{\mathbf{a}}(t) - \mathbf{b}_{a}(t) - \boldsymbol{\eta}_{a}(t)
\end{cases}$$

+ $\tilde{\boldsymbol{\omega}}, \tilde{\mathbf{a}}$：原始读数；$\boldsymbol{\eta}_{\omega}, \boldsymbol{\eta}_{a}$：高斯白噪声（方差$\sigma_\omega^2, \sigma_a^2$）；
+ 假设短时间内（k→k+1）偏置$\mathbf{b}_{\omega}, \mathbf{b}_a$为常数（缓慢时变特性）。

## 三、连续时间预积分推导（核心理论）

预积分的核心是 “消除绝对状态依赖”，通过定义相对状态，将绝对积分方程转化为相对增量方程。

### 1. 相对旋转预积分（$\Delta \mathbf{q}_{k+1}^k$）

理想情况下：
$$
R_{b_k}^{b_{k+1}} =exp(∫_{t_k}^{t_{k+1}}[ω_t^{true}]×dt)
$$

实际中，IMU 的角速率输出受偏差和噪声干扰，真实角速率需修正：

$$
ω_t^{true} = ω_t^b−b_ω
$$​

所以真实的旋转预积分

$$
Δ \widehat{R}_{k,k+1}(b_ω)=exp([Δ\widehat{θ}_{k,k+1}(b_ω)])
$$

### 2. 相对速度预积分（$\Delta \mathbf{v}_{k+1}^k$）

$$
{\Delta \mathbf{v}}(t)_k = \Delta \mathbf{R_k}(t) \mathbf{a}(t) - \mathbf{g}(k)
$$

## 四、离散化实现（工程落地关键）

IMU 是离散采样（采样周期$\Delta t$），需将连续方程离散化。工程中首选中值积分（精度高于欧拉积分，兼顾速度与精度），假设 k→k+1 之间有 m 个 IMU 采样点（$t_i = k + i\Delta t$，$i=0,1,...,m$）。

### 1. 离散化相对旋转（$\Delta \mathbf{q}_{k+1}^k$）

递推公式：

$$
\Delta\tilde{R}_{k,k+1} = \prod_{i=k}^{k+1-1} \text{Rodrigues}\left( (\omega_{t_i}^b - \mathbf{b}_\omega) \cdot \Delta t \right)
$$

### 2. 离散化相对速度（$\Delta \mathbf{v}_{k+1}^k$）

递推公式：
$$
\Delta \mathbf{v}_{i+1}^k = \Delta \mathbf{v}_i^k + \left( \frac{\Delta \mathbf{R}_i^k + \Delta \mathbf{R}_{i+1}^k}{2} \cdot \mathbf{a}_i^b - \mathbf{g}^b(k) \right) \cdot \Delta t
$$

其中：

+ $\mathbf{a}_i^b = \frac{\tilde{\mathbf{a}}_i^b + \tilde{\mathbf{a}}_{i+1}^b}{2} - \mathbf{b}_a$（中值去偏加速度）；
+ $\Delta \mathbf{R}_i^k$是$\Delta \mathbf{q}_i^k$对应的旋转矩阵。

最终 k→k+1 的相对速度：$\Delta \mathbf{v}_{k+1}^k = \Delta \mathbf{v}_m^k$。

### 3. 离散化相对位置（$\Delta \mathbf{p}_{k+1}^k$）

递推公式：
$$
\Delta \mathbf{p}_{i+1}^k = \Delta \mathbf{p}_i^k + \Delta \mathbf{v}_i^k \cdot \Delta t + \frac{1}{2} \left( \frac{\Delta \mathbf{R}_i^k + \Delta \mathbf{R}_{i+1}^k}{2} \cdot \mathbf{a}_i^b - \mathbf{g}^b(k) \right) \cdot \Delta t^2
$$

最终 k→k+1 的相对位置：$\Delta \mathbf{p}_{k+1}^k = \Delta \mathbf{p}_m^k$。

## 五、雅可比矩阵与偏置更新（优化核心）

后端优化会调整偏置（$\mathbf{b} \to \mathbf{b} + \delta \mathbf{b}$），为了避免重新积分 IMU 数据，需计算预积分因子对偏置的雅可比矩阵，通过一阶泰勒展开快速修正预积分结果。

### 1. 雅可比矩阵定义

需计算 6 个雅可比（对应陀螺偏置 $\mathbf{b}_\omega$ 和加计偏置 $\mathbf{b}_a$）：

$$
\begin{cases}
J_{qb\omega} = \frac{\partial \Delta \mathbf{q}_{k+1}^k}{\partial \mathbf{b}_\omega} \quad (4 \times 3) \\
J_{vb\omega} = \frac{\partial \Delta \mathbf{v}_{k+1}^k}{\partial \mathbf{b}_\omega}, \quad J_{vba} = \frac{\partial \Delta \mathbf{v}_{k+1}^k}{\partial \mathbf{b}_a} \quad (3 \times 3) \\
J_{pb\omega} = \frac{\partial \Delta \mathbf{p}_{k+1}^k}{\partial \mathbf{b}_\omega}, \quad J_{pba} = \frac{\partial \Delta \mathbf{p}_{k+1}^k}{\partial \mathbf{b}_a} \quad (3 \times 3)
\end{cases}
$$

### 2. 雅可比递推公式（以中值积分为例）

雅可比的核心是 “递推计算”，利用单步雅可比推导全局雅可比，以下给出关键递推关系（详细求偏导过程略，核心是链式法则）：

#### （1）旋转雅可比 $J_{qb\omega}$

$J_{q,i+1} = J_{q,i} + \frac{1}{2} \Delta \mathbf{q}_i^k \otimes (\boldsymbol{\omega}_i^b)^\wedge \cdot J_{q,i} - \frac{1}{2} \Delta \mathbf{q}_{i+1}^i \cdot \mathbf{J}_{\Delta q}$

+ 初始条件：$J_{q,0} = \mathbf{0}_{4 \times 3}$；
+ $\mathbf{J}_{\Delta q}$ 是单步旋转对偏置的雅可比（常数矩阵）。

#### （2）速度雅可比$J_{vb\omega}, J_{vba}$

$$
\begin{cases}
J_{vb\omega,i+1} = J_{vb\omega,i} + \frac{1}{2} \left( \mathbf{J}_{Rq,i} \cdot J_{q,i} + \mathbf{J}_{Rq,i+1} \cdot J_{q,i+1} \right) \cdot \mathbf{a}_i^b \cdot \Delta t \\
J_{vba,i+1} = J_{vba,i} - \frac{1}{2} (\Delta \mathbf{R}_i^k + \Delta \mathbf{R}_{i+1}^k) \cdot \Delta t
\end{cases}
$$

+ 初始条件：$J_{vb\omega,0} = \mathbf{0}_{3 \times 3}, J_{vba,0} = \mathbf{0}_{3 \times 3}$；
+ $\mathbf{J}_{Rq} = \frac{\partial \Delta \mathbf{R}}{\partial \Delta \mathbf{q}}$（旋转矩阵对四元数的雅可比，3×4）。

#### （3）位置雅可比$J_{pb\omega}, J_{pba}$

$$
\begin{cases}
J_{pb\omega,i+1} = J_{pb\omega,i} + J_{vb\omega,i} \cdot \Delta t + \frac{1}{4} \left( \mathbf{J}_{Rq,i} \cdot J_{q,i} + \mathbf{J}_{Rq,i+1} \cdot J_{q,i+1} \right) \cdot \mathbf{a}_i^b \cdot \Delta t^2 \\
J_{pba,i+1} = J_{pba,i} + J_{vba,i} \cdot \Delta t - \frac{1}{4} (\Delta \mathbf{R}_i^k + \Delta \mathbf{R}_{i+1}^k) \cdot \Delta t^2
\end{cases}
$$

+ 初始条件：$J_{pb\omega,0} = \mathbf{0}_{3 \times 3}, J_{pba,0} = \mathbf{0}_{3 \times 3}$。

### 3. 偏置更新后的预积分修正

当偏置从 $\mathbf{b}$ 更新为 $\mathbf{b}' = \mathbf{b} + \delta \mathbf{b}$ 时，预积分因子通过一阶泰勒展开修正：

$$
\boxed{
\begin{cases}
\Delta \mathbf{q}' \approx \Delta \mathbf{q} \otimes \left( \frac{1}{2} J_{qb\omega} \delta \mathbf{b}_\omega \right)^\wedge \\
\Delta \mathbf{v}' \approx \Delta \mathbf{v} + J_{vb\omega} \delta \mathbf{b}_\omega + J_{vba} \delta \mathbf{b}_a \\
\Delta \mathbf{p}' \approx \Delta \mathbf{p} + J_{pb\omega} \delta \mathbf{b}_\omega + J_{pba} \delta \mathbf{b}_a
\end{cases}
}
$$

工程价值：无需重新积分，毫秒级完成预积分修正，支撑后端实时优化。

## 六、预积分在后端优化中的应用

预积分的最终产物是 “相对运动约束”，在因子图优化中，通过 “绝对状态与预积分因子的一致性” 构建误差项，纳入全局优化目标。

### 1. 误差项定义

设 k 帧绝对状态为 $\mathbf{X}_k = (\mathbf{q}_k^w, \mathbf{v}_k^w, \mathbf{p}_k^w)$，k+1 帧为$\mathbf{X}_{k+1}$，预积分因子为$\Delta \mathbf{X}_{k+1}^k$，则误差项为：

$$
\boxed{
\begin{cases}
\mathbf{e}_q = \Delta \mathbf{q}_{k+1}^k \otimes (\mathbf{q}_{k+1}^w)^* \otimes \mathbf{q}_k^w \quad (\text{旋转误差，3维，四元数转旋转向量}) \\
\mathbf{e}_v = \mathbf{R}_k^w (\mathbf{v}_{k+1}^w - \mathbf{v}_k^w) - \Delta \mathbf{v}_{k+1}^k \quad (\text{速度误差，3维}) \\
\mathbf{e}_p = \mathbf{R}_k^w (\mathbf{p}_{k+1}^w - \mathbf{p}_k^w - \mathbf{v}_k^w \Delta t) - \Delta \mathbf{p}_{k+1}^k \quad (\text{位置误差，3维})
\end{cases}
}
$$

误差项的物理意义：“绝对状态推导的相对运动” 与 “预积分得到的相对运动” 的差值，理想情况下误差为 0。

### 2. 优化目标函数

全局优化的目标是最小化所有误差项的加权和（加权矩阵为误差协方差的逆）：

$$
\text{Cost} = \sum_{k=0}^{n-1} \left( \mathbf{e}_q^T \Sigma_q^{-1} \mathbf{e}_q + \mathbf{e}_v^T \Sigma_v^{-1} \mathbf{e}_v + \mathbf{e}_p^T \Sigma_p^{-1} \mathbf{e}_p \right)
$$

其中$\Sigma_q, \Sigma_v, \Sigma_p$是预积分噪声协方差（由 IMU 噪声和积分过程传播得到）。

### 3. 优化求解

通过 LM（Levenberg-Marquardt）算法最小化 Cost 函数，求解最优的绝对状态$\mathbf{X}_k$和偏置$\mathbf{b}_k$。误差项对状态的雅可比（如$\frac{\partial \mathbf{e}_q}{\partial \mathbf{q}_k^w}$）可通过链式法则推导，最终形成增量方程$J^T \Sigma^{-1} J \delta x = J^T \Sigma^{-1} e$，迭代求解增量$\delta x$，更新状态直至收敛。

## 七、总结

IMU 预积分的核心逻辑可概括为：

1. 动机：解决传统绝对积分的 “连锁更新” 和 “复用性差” 问题，适配后端实时优化；
2. 核心：将绝对积分转化为 “相对增量积分”，消除绝对状态依赖，仅依赖 IMU 读数和偏置；
3. 实现：通过中值积分完成离散化，通过雅可比递推支持偏置快速更新；
4. 应用：输出相对运动约束，纳入因子图优化，与激光、视觉等传感器约束联合求解最优状态。

预积分是紧耦合多传感器融合的 “基石”—— 正是因为预积分的存在，IMU 的高频信息才能被高效、精准地融入后端优化，最终实现高精度、高鲁棒性的位姿估计（如 LIO-SAM、FAST-LIO 等算法的核心就是 IMU 预积分因子）。
