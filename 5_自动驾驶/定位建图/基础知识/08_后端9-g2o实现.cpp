#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

// G2O 顶点：2D位姿
class VertexPose2D : public g2o::BaseVertex<3, Eigen::Vector3d> /*表示优化变量的维度是3，数据类型是Eigen::Vector3d*/{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    virtual void setToOriginImpl() override {
        _estimate.setZero();        // 初始化为零
    }
    
    virtual void oplusImpl(const double* update) override {
        _estimate[0] += update[0];  // x += Δx
        _estimate[1] += update[1];  // y += Δy
        _estimate[2] += update[2];  // θ += Δθ
    }
    
    virtual bool read(std::istream& is) override { return true; }
    virtual bool write(std::ostream& os) const override { return true; }
};

// G2O 边：点到直线距离残差
class EdgePointToLine : public g2o::BaseUnaryEdge<1, double, VertexPose2D> /*误差维度为1(点到直线距离)， 测量值的类型，链接的顶点类型*/ {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EdgePointToLine(const Line2D& line) : line_(line) {}
    
    virtual void computeError() override {
        const VertexPose2D* v = static_cast<const VertexPose2D*>(_vertices[0]);
        const Eigen::Vector3d& pose = v->estimate();            // 获取当前位姿
        
        // 计算有符号距离： n·point + d = 0 的偏差
        Eigen::Vector2d point(pose[0], pose[1]);
        double signed_distance = line_.n.dot(point) + line_.d;
        
        _error[0] = signed_distance;                            // 误差 = 实际距离
    }
    
    virtual void linearizeOplus() override {
        const VertexPose2D* v = static_cast<const VertexPose2D*>(_vertices[0]);
        const Eigen::Vector3d& pose = v->estimate();
        
        // 雅可比矩阵计算
        Eigen::Matrix<double, 1, 3> jacobian;
        jacobian << line_.n.x(), line_.n.y(), 0.0;  // 对x,y求导，对theta导数为0
        
        _jacobianOplusXi = jacobian;
    }
    
    virtual bool read(std::istream& is) override { return true; }
    virtual bool write(std::ostream& os) const override { return true; }

private:
    Line2D line_;
};

// G2O 优化函数
Pose2D optimizeWithG2O(const Pose2D& initial_pose, const Line2D& line) {
    // 创建优化器
    g2o::SparseOptimizer optimizer;
    
    // 配置求解器 使用 Levenberg-Marquardt优化算法
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolverX>(
            std::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
    optimizer.setAlgorithm(solver);
    
    // 1. 添加顶点（待优化的位姿）
    VertexPose2D* vertex = new VertexPose2D();
    vertex->setId(0);
    vertex->setEstimate(Eigen::Vector3d(initial_pose.x, initial_pose.y, initial_pose.theta));
    optimizer.addVertex(vertex);
    
    // 2. 添加边（约束条件）
    EdgePointToLine* edge = new EdgePointToLine(line);
    edge->setId(0);
    edge->setVertex(0, vertex);
    edge->setMeasurement(0.0);  // 理想距离为0 目标：点到直线距离为0
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());  // 信息矩阵
    optimizer.addEdge(edge);
    
    // 执行优化
    optimizer.initializeOptimization();
    optimizer.setVerbose(false);
    optimizer.optimize(10);
    
    // 获取结果
    Eigen::Vector3d result = vertex->estimate();
    return Pose2D(result[0], result[1], result[2]);
}