
# 后端4 因子图优化

接下来举三个例子来解释图优化和因子图优化

## 简单的移动

假设有一个机器人在二维平面上移动，有以下信息

1. 初始位姿 y0 (x, y, r) = (0, 0, 0) 也就是坐标和朝向
2. 里程计测量：机器人向北移动了3个单位，到达y1，又向东移动了 4 个单位，到达了 y2
3. 回环检测: 机器人在y1位置观测 y0 得到相对位置为 dy (0.2, -3.1, 0), 在 y2 观测 y0 得到的相对位置为 dy' (-4.1, -3.1, 0)

在这个运动里面，假设移动的角度绝对准确，其他所有的参数都或多或少有误差，那么我们如何通过图优化来解决这个问题呢？

这里面我们要确定准确的位姿 y0，y1，y2

### 图优化建模

#### 首先构建图模型，其中节点表示位姿，边表示里程计测量和回环检测

+ 节点: y0，y1，y2
+ 边（也就是各种约束）:
  + 先验约束:也就是初始位姿也不确定 （固定x0的初始值（0, 0））
  + 里程计约束：约束相邻位姿间的位移
    + y0 -> y1: 想北移动了3单位
    + y1 -> y2: 想向东移动了4单位
  + 回环约束：约束位姿间的相对位置
    + 在 y1 观测到 y0 的相对位置 (0.2, -3.1)
    + 在 y2 观测到 y0 的相对位置 (4.1, -3.1)

#### 确定协方差矩阵

不同的传感器的噪声特点不同，需要合理分配权重

#### 确定残差定义

1. 先验残差
  $residual = y_{0估计} - y_{0先验} = [y0.x - 0, y0.y - 0]$
2. 里程计残差
  $residual = Dis_{估计} - Dis_{后验}$
     + y0 -> y1: $residual = [y1.x - y0.x - 0, y1.y - 3]$
     + y1 -> y2: $residual = [y2.x - y1.x - 4, y2.y - y1.y]$
3. 回环残差
  $residual = P_{观测值} - P_{真实值}$
     + 在y1 处 $residual = [0.2 - (y1.x - y0.x), -3.1-(y1.x-y0.x)]$
     + 在y2 处 $residual = [4.1 - (y2.x - y0.x), -3.1-(y2.x-y0.x)]$

#### 目标函数

这样问题就转化成最小二乘问题
$$
\min_{y1, y2, y3} \sum_{i=1}^{n} r^Tr \sum ^{-1} = \min_{y1, y2, y3} \sum||residual_{先验}||^2 + \sum||residual_{里程计}||^2 + \sum||residual_{回环}||^2
$$

其中 $\sum ^{-1}$ 是协方差矩阵的逆矩阵, 作为各个因子的权重，称为信息矩阵

#### 代码实现

```cpp
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen:

// 定义先验误差
struct PriorResidual {
    PriorResidual(double x_prior, double y_prior, double sigmal)
        : x_prior_(x_prior), y_prior_(y_prior), sigma_(sigmal) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residual) const {
        residual[0] = (pose[0] - T(x_prior_)) / T(sigma_);
        residual[1] = (pose[1] - T(y_prior_)) / T(sigma_);
        return true;
    }
private:
    double x_prior_, y_prior_, sigma_;
};

// 定义观测残差 (不考虑运动方向)
struct ObservationResidual {
    ObservationResidual(double x_obs, double y_obs, double sigma) : x_obs_(x_obs), y_obs_(y_obs), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residual) const {
        T dx = pose_j[0] - pose_i[0];
        T dy = pose_j[1] - pose_i[1];
        residual[0] = (dx - T(x_obs_)) / T(sigma_);
        residual[1] = (dy - T(y_obs_)) / T(sigma_);
        return true;
    }
private:
    double x_obs_, y_obs_, sigma_;
};

// 定义里程计残差 (考虑协方差)
struct OdometryResidual {
    OdometryResidual(double x_odom, double y_odom, double sigma) : x_odom_(x_odom), y_odom_(y_odom), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residual) const {
        residual[0] = (pose_j[0] - pose_i[0] - T(x_odom_)) / T(sigma_);
        residual[1] = (pose_j[1] - pose_i[1] - T(y_odom_)) / T(sigma_);
        return true;
    }

private:
    double x_odom_, y_odom_, sigma_;
};

int main() {
    // 初始化位姿
    double y0[2] = {0.0, 0.0};
    double y1[2] = {0.0, 3.0};
    double y2[2] = {4.0, 3.0};

    // 创建优化问题
    ceres::Problem problem;

    // 添加先验因子 (固定 y0)
    ceres::CostFunction* prior_cost = new ceres::AutoDiffCostFunction<PriorResidual, 2, 2>(new PriorResidual(0.0, 0.0, 0.1));
    problem.AddResidualBlock(prior_cost, nullptr, y0);

    // 添加里程计因子
    ceres::CostFunction* odom_cost1 = new ceres::AutoDiffCostFunction<OdometryResidual, 2, 2, 2>(new OdometryResidual(0.0, 3.0, 0.5));
    problem.AddResidualBlock(odom_cost1, nullptr, y0, y1);

    ceres::CostFunction* odom_cost2 = new ceres::AutoDiffCostFunction<OdometryResidual, 2, 2, 2>(new OdometryResidual(4.0, 0.0, 0.5));
    problem.AddResidualBlock(odom_cost2, nullptr, y1, y2);

    // 添加观测因子
    ceres::CostFunction* observation_cost1 = new ceres::AutoDiffCostFunction<ObservationResidual, 2, 2, 2>(new ObservationResidual(0.2, -3.1, 0.3));
    problem.AddResidualBlock(observation_cost1, nullptr, y0, y1);

    ceres::CostFunction* observation_cost2 = new ceres::AutoDiffCostFunction<ObservationResidual, 2, 2, 2>(new ObservationResidual(4.1, -3.1, 0.3));

    // 匹配并运行优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    std::cout << "Optimized y0: (" << y0[0] << ", " << y0[1] << ")\n";
    std::cout << "Optimized y1: (" << y1[0] << ", " << y1[1] << ")\n";
    std::cout << "Optimized y2: (" << y2[0] << ", " << y2[1] << ")\n";

    return 0;
}
```

#### 函数参数介绍

`ceres::AutoDiffCostFunction<PriorResidual, 2, 2>(new PriorResidual(0.0, 0.0, 0.1));` 定义了观测点的移动，其中 `PriorResidual` 是观测点的先验误差，`ObservationResidual` 是观测点的观测误差，`OdometryResidual` 是里程计的误差。

+ 第一个参数 CostFunctor： 用户自定义的残差计算类
+ 第二个参数 kNumResiduals: 残差的维度（即残差向量的长度）
+ 后续参数：各个参数块的维度，（每个参数块对应优化变量的一部分）


`ceres::Solver::Options options;` 常见的选项包括线性求解器类型、最大迭代次数、最大时间、进度输出、梯度检查等。此外，还有关于收敛条件、参数块排序、线程数等设置。

+ 线性求解器配置：控制底层线性代数求解器的行为，直接影响优化速度和稳定性。
  + linear_solver_type
    + DENSE_QR(默认): 使用稠密QR分解求解线性方程组，适用于参数小于1000的问题
    + DENSE_NORMAL_CHOLESKY: 使用稠密正则化Cholesky分解求解线性方程组，适用于大型稠密问题
    + SPARSE_NORMAL_CHOLESKY: 使用稀疏正则化Cholesky分解求解线性方程组，适用于稀疏问题
    + CGNR: 使用共轭梯度法求解线性方程组
  + preconditioner_type:
    + IDENTITY: 不使用预处理
    + JACOBI: 使用雅可比矩阵的对角线元素作为预处理
    + SCHUR_JACOBI(默认): 使用舒尔补的对角线元素作为预处理
    + CLUSTER_JACOBI: 使用聚类的对角线元素作为预处理
+ 收敛条件：控制优化何时停止
  + max_num_iterations: 最大迭代次数
    + 默认值： 50
  + max_solver_time_in_seconds: 最大求解时间（秒）默认无限
  + function_tolerance: 函数值的容忍度，目标函数的变化下雨这个值就会停止优化
  + parameter_tolerance: 参数值的容忍度，参数的变化小于这个值就会停止优化，通常和函数值的容忍度设置为相同的量级
+ 输出和调试：控制优化过程中的输出信息和日志
  + minimizer_progress_to_stdout:
    + 默认值 false，设为 true 时，在控制台打印每次迭代的进度，包括残差和耗时等
  + logging_type: 控制日志详细程度
    + 可选值 SLIENT，PER_MINIMIZER_ITERATION
  + check_gradients
    + 默认值 false。置为true时验证用户提供的雅可比矩阵是否正确，仅在调试时使用

#### g2o 代码实现

g2o（General Graph Optimization）是一个​​开源图优化库​​，专门用于求解基于图模型的非线性最小二乘问题。在自动驾驶中，g2o的核心应用场景是​​状态估计​​和​​传感器融合​​。

g2o的核心功能​​

+ ​求解位姿图优化​​：例如SLAM中的回环检测后，优化全局轨迹和地图的一致性。
+ ​多传感器联合标定​​：例如相机-激光雷达外参标定。
+ ​运动补偿与滤波​​：例如融合IMU与视觉数据，补偿运动模糊。

具体应用场景​

+ ​激光SLAM​​
  + 在LOAM、LeGO-LOAM等算法中，g2o用于优化激光雷达扫描匹配后的位姿图
+ 视觉惯性里程计（VIO）​​：
  + 融合相机和IMU数据，g2o用于优化状态变量（位置、速度、零偏、重力方向）。
+ 在线定位与地图更新​​：
  + 在动态环境中，g2o实时优化局部地图与全局地图的对齐关系。

为什么自动驾驶项目要求g2o？​​
+ ​高效稀疏求解​​：g2o针对SLAM问题中典型的稀疏雅可比矩阵优化，采用舒尔补（Schur Complement）等技术加速求解。
+ ​灵活的因子定义​​：支持自定义边（Edge）实现新的传感器约束（例如轮速计、GPS）。
+ ​社区与工业支持​​：ORB-SLAM、Cartographer等经典算法依赖g2o，生态成熟。

```cpp
#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;
using namespace g2o;

// 自定义二维顶点（x, y）
class VertexPointXY : public BaseVertex<2, Vector2d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPointXY() {}

    virtual void setToOriginImpl() override {
        _estimate.setZero();
    }

    virtual void oplusImpl(const double* update) override {
        _estimate += Eigen::Map<const Vector2d>(update);
    }

    virtual bool read(istream& in) { return false; }
    virtual bool write(ostream& out) const { return false; }
};

// 先验边（固定y0的初始位置）,继承基础一元边
class PriorEdge : public BaseUnaryEdge<2, Vector2d, VertexPointXY> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PriorEdge(const Vector2d& measurement, const Matrix2d& info_matrix)
        : _measurement(measurement) {
        setInformation(info_matrix);
    }

    virtual void computeError() override {
        const VertexPointXY* v = static_cast<const VertexPointXY*>(_vertices[0]);
        _error = v->estimate() - _measurement;
    }

private:
    Vector2d _measurement;
};

// 里程计边（位移约束）
class OdometryEdge : public BaseBinaryEdge<2, Vector2d, VertexPointXY, VertexPointXY> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    OdometryEdge(const Vector2d& measurement, const Matrix2d& info_matrix)
        : _measurement(measurement) {
        setInformation(info_matrix);
    }

    virtual void computeError() override {
        const VertexPointXY* v1 = static_cast<const VertexPointXY*>(_vertices[0]);
        const VertexPointXY* v2 = static_cast<const VertexPointXY*>(_vertices[1]);
        _error = (v2->estimate() - v1->estimate()) - _measurement;
    }

private:
    Vector2d _measurement;
};

// 观测边（考虑运动方向）
class ObservationEdge : public BaseBinaryEdge<2, Vector2d, VertexPointXY, VertexPointXY> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ObservationEdge(const Vector2d& measurement, const Matrix2d& info_matrix, double theta)
        : _measurement(measurement), _theta(theta) {
        setInformation(info_matrix);
    }

    virtual void computeError() override {
        const VertexPointXY* v_i = static_cast<const VertexPointXY*>(_vertices[0]); // y0
        const VertexPointXY* v_j = static_cast<const VertexPointXY*>(_vertices[1]); // y1/y2

        // 将局部观测转换到世界坐标系
        Matrix2d R;
        R << cos(_theta), -sin(_theta), sin(_theta), cos(_theta);
        Vector2d p_obs_world = R * _measurement + v_j->estimate();

        _error = v_i->estimate() - p_obs_world;
    }

private:
    Vector2d _measurement;
    double _theta; // 运动方向（弧度）
};

int main() {
    // 创建优化器
    typedef BlockSolver<BlockSolverTraits<2, 2>> BlockSolverType;
    typedef LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 添加顶点
    VertexPointXY* v0 = new VertexPointXY();
    VertexPointXY* v1 = new VertexPointXY();
    VertexPointXY* v2 = new VertexPointXY();
    v0->setId(0);
    v1->setId(1);
    v2->setId(2);
    v0->setEstimate(Vector2d(0, 0));   // 初始值
    v1->setEstimate(Vector2d(0, 3));   // 里程计测量：北移3
    v2->setEstimate(Vector2d(4, 3));   // 里程计测量：东移4
    optimizer.addVertex(v0);
    optimizer.addVertex(v1);
    optimizer.addVertex(v2);

    // 添加先验边（固定y0）
    Matrix2d info_prior = Matrix2d::Identity() / (0.1 * 0.1); // 信息矩阵 = 1/σ²
    PriorEdge* prior_edge = new PriorEdge(Vector2d(0, 0), info_prior);
    prior_edge->setVertex(0, v0);
    optimizer.addEdge(prior_edge);

    // 添加里程计边
    // y0 -> y1 北移3单位
    Matrix2d info_odom = Matrix2d::Identity() / (0.5 * 0.5);
    OdometryEdge* odom_y0_y1 = new OdometryEdge(Vector2d(0, 3), info_odom);
    odom_y0_y1->setVertex(0, v0);
    odom_y0_y1->setVertex(1, v1);
    optimizer.addEdge(odom_y0_y1);

    // y1 -> y2 东移4单位
    OdometryEdge* odom_y1_y2 = new OdometryEdge(Vector2d(4, 0), info_odom);
    odom_y1_y2->setVertex(0, v1);
    odom_y1_y2->setVertex(1, v2);
    optimizer.addEdge(odom_y1_y2);

    // 添加观测边
    Matrix2d info_obs = Matrix2d::Identity() / (0.2 * 0.2);
    // 在y1（朝北，theta=90度）观测y0
    ObservationEdge* obs_y1 = new ObservationEdge(Vector2d(0.2, -3.1), info_obs, M_PI/2);
    obs_y1->setVertex(0, v0); // y0
    obs_y1->setVertex(1, v1); // y1
    optimizer.addEdge(obs_y1);

    // 在y2（朝东，theta=0度）观测y0
    ObservationEdge* obs_y2 = new ObservationEdge(Vector2d(4.1, -3.1), info_obs, 0);
    obs_y2->setVertex(0, v0); // y0
    obs_y2->setVertex(1, v2); // y2
    optimizer.addEdge(obs_y2);

    // 执行优化
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(10);

    // 输出结果
    cout << "Optimized y0: (" << v0->estimate().x() << ", " << v0->estimate().y() << ")\n";
    cout << "Optimized y1: (" << v1->estimate().x() << ", " << v1->estimate().y() << ")\n";
    cout << "Optimized y2: (" << v2->estimate().x() << ", " << v2->estimate().y() << ")\n";

    return 0;
}
```

#### 鲁棒核函数

在因子图优化中，异常值（如错误的观测或里程计数据）会导致优化目标被污染，传统的最小二乘法会过度放大这些异常值的影响。​​鲁棒核函数（Robust Kernel）​​通过对大残差进行抑制，降低异常值的权重，从而提高系统的鲁棒性。

常见的鲁棒核函数包括：

+ ​​Huber​​：对小于阈值δ的残差使用平方损失，大于δ的残差使用线性损失。
+ Cauchy: 对残差进行对数加权，抑制大残差的影响。
+ Tukey: 对残差进行三次加权, 对大残差完全截断，适用于严格剔除异常值。

以下是常见的鲁棒核函数（Robust Kernel）的数学公式及其作用，它们通过修改损失函数（Loss Function）来抑制异常值的影响。

---

### **1. Huber 核函数**
• **公式**：
$$
  \rho(r) = \begin{cases} 
  \frac{1}{2} r^2 & \text{if } |r| \leq \delta \\
  \delta \left(|r| - \frac{1}{2} \delta \right) & \text{if } |r| > \delta 
  \end{cases}
$$
+ **参数**：阈值 \(\delta\)，控制线性损失和平方损失的切换点。
+ **特点**：
  + 对小残差（\(|r| \leq \delta\)）保持平方损失，维持高斯假设。
  + 对大残差（\(|r| > \delta\)）转为线性损失，降低异常值的梯度。
+ **适用场景**：通用场景，平衡计算效率和鲁棒性。

---

### **2. Cauchy 核函数**
• **公式**：
$$
  \rho(r) = \frac{c^2}{2} \log\left(1 + \left(\frac{r}{c}\right)^2\right)
$$
+ **参数**：比例因子 \(c\)，控制加权强度。
+ **特点**：
  + 对所有残差进行对数加权，对大残差梯度衰减更快。
  + 比Huber核更激进地抑制异常值。
+ **适用场景**：存在较多离群点的场景（如动态环境中的视觉SLAM）。

---

### **3. Tukey（双权）核函数**
• **公式**：
$$
  \rho(r) = \begin{cases} 
  \frac{c^2}{6} \left[1 - \left(1 - \left(\frac{r}{c}\right)^2\right)^3
  \right] & |r| \leq c \\
  \frac{c^2}{6} & |r| > c 
  \end{cases}
$$

+ **参数**：截断阈值 \(c\)，超出阈值的残差梯度直接置零。
+ **特点**：
  + 对超出阈值的残差完全截断（梯度为0），彻底剔除异常值。
  + 需要精确设置阈值 \(c\)，否则可能误删有效数据。
+ **适用场景**：严格剔除异常值（如已知传感器最大误差范围）。

---

### **4. Geman-McClure 核函数**
• **公式**：
  $$
  \rho(r) = \frac{r^2}{2 \left(1 + r^2 \right)}
  $$
+ **参数**：无显式参数，但需注意残差标准化。
+ **特点**：
  + 对大残差的梯度衰减更剧烈（随 \(r^2\) 的四次方衰减）。
  + 在视觉SLAM中常用于重投影误差优化。
+ **适用场景**：需要强鲁棒性的稠密优化问题。

---

### **5. Welsh 核函数**
• **公式**：
  $$
  \rho(r) = \frac{c^2}{2} \left(1 - e^{-\frac{r^2}{c^2}} \right)
  $$
+ **参数**：衰减系数 \(c\)，控制指数衰减速度。
+ **特点**：
  + 对大残差梯度指数衰减，平滑抑制异常值。
  + 对高斯噪声保留较好的统计特性。
+ **适用场景**：传感器噪声接近高斯分布但有少量离群点。

---

### **6. 对比与选择指南**
| **核函数**       | **公式特点**                          | **优点**                      | **缺点**                      |
|------------------|---------------------------------------|-------------------------------|-------------------------------|
| **Huber**        | 分段线性+平方损失                     | 计算高效，通用性强             | 需要手动设置阈值 \(\delta\)    |
| **Cauchy**       | 对数加权                              | 强鲁棒性，适合动态环境         | 可能过度抑制大残差             |
| **Tukey**        | 截断阈值                              | 彻底剔除异常值                 | 阈值设置敏感，易误删有效数据   |
| **Geman-McClure**| 梯度随 \(r^4\) 衰减                   | 对离群点敏感                   | 无显式参数，灵活性低           |
| **Welsh**        | 指数衰减                              | 平滑抑制，保留高斯特性          | 需要标准化残差                |

---

### **在 Ceres Solver 中的使用示例**
```cpp
// 添加残差块时指定鲁棒核函数
problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<MyCostFunctor, 1, 1>(
        new MyCostFunctor(...)
    ),
    new ceres::CauchyLoss(0.5), // Cauchy核，c=0.5
    parameter_block
);
```
