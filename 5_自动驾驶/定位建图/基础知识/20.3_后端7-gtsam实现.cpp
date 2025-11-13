#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/PriorFactor.h>

// GTSAM 自定义因子：点到直线距离
class PointToLineFactor : public gtsam::NoiseModelFactor1<gtsam::Pose2> {
public:
    PointToLineFactor(gtsam::Key poseKey, const Line2D& line, 
                     const gtsam::SharedNoiseModel& model) 
        : gtsam::NoiseModelFactor1<gtsam::Pose2>(model, poseKey), line_(line) {}
    
    gtsam::Vector evaluateError(const gtsam::Pose2& pose,
                               boost::optional<gtsam::Matrix&> H = boost::none) const override {
        // 提取位姿中的点
        gtsam::Point2 point = pose.translation();
        
        // 计算有符号距离
        double signed_distance = line_.n.x() * point.x() + line_.n.y() * point.y() + line_.d;
        
        // 计算雅可比矩阵（如果需要）
        if (H) {
            *H = gtsam::Matrix::Zero(1, 3);
            (*H)(0, 0) = line_.n.x();  // 对x的导数
            (*H)(0, 1) = line_.n.y();  // 对y的导数
            (*H)(0, 2) = 0.0;          // 对theta的导数
        }
        
        return gtsam::Vector1(signed_distance);
    }
    
private:
    Line2D line_;
};

// GTSAM 优化函数
Pose2D optimizeWithGTSAM(const Pose2D& initial_pose, const Line2D& line) {
    // 创建因子图
    gtsam::NonlinearFactorGraph graph;
    
    // 添加噪声模型
    auto noise_model = gtsam::noiseModel::Isotropic::Sigma(1, 0.1);
    
    // 添加自定义因子
    gtsam::Symbol poseKey('x', 0);
    graph.add(std::make_shared<PointToLineFactor>(poseKey, line, noise_model));
    
    // 添加位姿先验（可选，帮助优化收敛）
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        gtsam::Vector3(0.1, 0.1, 0.1));
    graph.add(gtsam::PriorFactor<gtsam::Pose2>(poseKey, 
        gtsam::Pose2(initial_pose.x, initial_pose.y, initial_pose.theta), prior_noise));
    
    // 设置初始值
    gtsam::Values initial_values;
    initial_values.insert(poseKey, gtsam::Pose2(initial_pose.x, initial_pose.y, initial_pose.theta));
    
    // 执行优化
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_values);
    gtsam::Values result = optimizer.optimize();
    
    // 提取结果
    gtsam::Pose2 optimized_pose = result.at<gtsam::Pose2>(poseKey);
    return Pose2D(optimized_pose.x(), optimized_pose.y(), optimized_pose.theta());
}