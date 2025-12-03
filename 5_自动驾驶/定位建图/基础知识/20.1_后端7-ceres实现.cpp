#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>

using namespace Eigen;
using namespace ceres;

// -------------------------- 数据结构定义 ------------------------
// 位姿状态：位置(x, y, z) + 姿态(四元数 w, x, y, z) + 速度(vx, vy, vz) + IMU零偏(bwx, bwy, bwz, bax, bay, baz)
struct PoseState {
    double t;                   // 时间戳
    Vector3d p;                 // 位置
    Quaterniond q;              // 姿态（单位四元数）
    Vector3d v;                 // 速度
    Vector3d b_w;               // 陀螺仪零偏
    Vector3d b_a;               // 加速度计零偏

    PoseState() : q(1, 0, 0, 0) {}
    PoseState(double _t, const Vector3d& _p, const Quaterniond& _q, const Vector3d& _v,
              const Vector3d& _bw, const Vector3d& _ba)
        : t(_t), p(_p), q(_q.normalized()), v(_v), b_w(_bw), b_a(_ba) {} 
};

// IMU 预积分结果 (预计算后传入残差，避免重复计算)
struct IMUPreintegration {
    double dt;                  // 预积分时间
    Quaterniond delta_q;        // 姿态增量
    Vector3d delta_p;           // 位置增量
    Vector3d delta_v;           // 速度增量
    Matrix<double, 15, 15> cov; // 预积分协方差
    Matrix<double, 15, 6> jacobian; // 雅可比矩阵，相对于零偏
    Vector3d gravity;           // 重力加速度（默认 9.81m/s2）
};

// 视觉特征观测（用于重投影误差）
struct VisualObservation {
    int cam_id;                 // 相机ID
    Vector2d uv;                // 图片平面观测坐标(u, v)
    Vector3d P_w;               // 特征点世界坐标
    Matrix2d info;              // 观测信息矩阵（协方差矩阵）
};

// GNSS 观测（绝对位置）
struct GNSSObservation {
    Vector3d p_w;               // GNSS测量的世界坐标
    Matrix3d info;              // 观测信息矩阵
};

// 轮速记观测（线速度+角速度）
struct WheelOdometryObservation {
    double v;                   // 测量线速度
    double w;                   // 测量角速度
    Matrix2d info;              // 观测信息矩阵
};

// 回环检测观测（两帧位姿约束）
struct LoopClosureObservation {
    int target_frame_id;        // 回环目标帧ID
    Quaterniond delta_q;        // 相对姿态约束
    Vector3d delta_p;           // 相对位置约束
    Matrix6d info;              // 观测信息矩阵
};

// -------------------------- 残差派生类实现 ------------------------
// 1. 视觉重投影误差残差（2维残差： u误差， v误差）
class ReprojectionError : public SizedCostFunction<2, 3, 4> {   // 2维残差，第一个参数3维，第二个参数4维
public:
    ReprojectionError(const VisualObservation& obs, const Matrix3d& K)
        : obs_(obs), K_(K) {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {
        // parameters[0]: 当前帧位置 p (x, y, z)
        // parameters[1]: 当前帧姿态 q (w, x, y, z)
        const Vector3d p(parameters[0][0], parameters[0][1], parameters[0][2]);
        const Quaterniond q(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]);

        // 世界点转换到相机坐标系：P_c = q * (P_w - p)
        const Vector3d P_c = q*(obs_.P_w - p);
        if (P_c.z() <= 0) {     // 特征点在相机后方为无效观测
            residuals[0] = residuals[1] = 1e6;
            return true;
        }

        // 相机坐标系投影到图像平面：uv = K*P_c / P_c.z()
        const Vector3d uv_proj = K_ * P_c / P_c.z();
        const Vector2d uv_pred(uv_proj.x(), uv_proj.y());

        // 残差计算： res = info^(1/2) * (观测 - 预测)
        const Vector2d res = obs_.uv - uv_pred;
        res
    }
private:
    VisualObservation obs_;
    Matrix3d K_;                // 相机内参矩阵

    // 辅助函数：反对称矩阵
    Matrix3d skew_symmetric(const Vector3d& v) const {
        Matrix3d mat;
        mat << 0, -v.z(), v.y(),
               v.z(), 0, -v.x(),
               -v.y(), v.x(), 0;
        return mat;
    }
};