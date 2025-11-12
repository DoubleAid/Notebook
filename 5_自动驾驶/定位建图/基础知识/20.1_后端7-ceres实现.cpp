#include <vector>
#include <iostream>
#include <Eigen/Dense>

// 2D位姿 (x, y, theta)
struct Pose2D {
    double x, y, theta;
    Pose2D(double x = 0, double y = 0, double theta = 0): x(x), y(y), theta(theta) {}
};

// 直线表示: n · p + d = 0 (法线形式)
struct Line2D {
    Eigen::Vector2d n;  // 单位法向量
    double d;           // 距离原点的有符号距离

    Line2D(const Eigen::Vector2d& normal, double distance) : n(normal.normalized()), d(distance) {}

    // 从两点构造直线
    static Line2D fromTwoPoints(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) {
        Eigen::Vector2d dir = (p2 - p1).normalized();
        Eigen::Vector2d normal(-dir.y(), dir.x());  // 垂直方向
        double distance = -normal.dot(p1);
        return Line2D(normal, distance);
    }
};

class PointToLineFactor : public ceres::SizedCostFunction<1 /*残差维度*/, 3 /*参数块维度*/> {
public:
    // 构造函数: 传入局部点坐标到直线参数
    PointToLineFactor(const Eigen::Vector2d& local_pt,
                      const Eigen::Vector2d& line_normal,
                      double line_d) : local_pt_(local_pt), line_normal_(line_normal), line_d_(line_d) {}

    // 核心函数： 计算残差和 (可选的) 雅可比矩阵
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        // 1. 解包参数 
        const double x      = parameters[0][0];
        const double y      = parameters[0][1];
        const double theta  = parameters[0][2];

        // 2. 计算旋转矩阵 R = [cosθ -sinθ \\ sinθ cosθ]
        const double cos_theta = cos(theta);
        const double sin_theta = sin(theta);

        // 3. 将局部点转换到世界坐标系 p_w = R * p_l + t
        Eigen::Vector2d p_w;
        p_w(0) = cos_theta * local_pt_(0) - sin_theta * local_pt_(1) + x;
        p_w(1) = sin_theta * locak_pt_(1) + cos_theta * local_pt_(1) + y;
    }
private:
    Eigen::Vector2d local_pt_;          // 局部坐标系下的点
    Eigen::Vector2d line_normal_;       // 直线的单位法向量
    double line_d_;              // 直线参数 d
};