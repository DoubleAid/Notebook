#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>

class ESKF {
public:
    ESKF();
    ~ESKF() = default

    // 配置传感器噪声参数（供ROS节点调用，实现松耦合）
    void set_noise_params(double imu_gyro_noise, double imu_acc_noise,
                          double odom_noise, double lidar_noise, double gps_noise);
    
    // IMU 预测（125Hz 高频）
    void predict(const sensor_msgs::msg::Imu::SharedPtr& imu_msg);
    
    // 轮速记更新 (10Hz)
    void update_odom();

    // 激光雷达更新（10Hz）
    void update_lidar(const Eigen::Vector3d& trans, const Eigen::Quaterniond& rot);

    // GPS 更新（1/5Hz）
    void update_gps(const Eigen::Vector3d& utm_p, int gps_quality);

    // 获取融合后的状态 (对外接口)
    Eigen::Vector3d get_position() const { return state_p_; }
    Eigen::Vector3d get_velocity() const { return state_v_; }
    Eigen::Quaterniond get_orientation() const { return state_q_; }

private:
    // 误差状态反馈到绝对状态并重置
    void feedback_and_reset();

    // 向量的反对称矩阵（辅助函数）
    Eigen::Matrix3d skew(const Eigen::Vector3d& v);

    // 绝对状态：位置，速度，姿态，IMU零偏
    Eigen::Vector3d state_p_;           // 位置 (m)
    Eigen::Vector3d state_v_;           // 速度 (m/s)
    Eigen::Quaterniond statq_q_;        // 姿态四元数
    Eigen::Vector3d state_b_gyro_;      // IMU角速度零偏 (rad/s)
    Eigen::Vector3d state_b_acc_;       // IMU加速度零偏 (m/s²)

    // 误差状态：15维（δp(3)+δv(3)+δθ(3)+δb_gyro(3)+δb_acc(3)）
    Eigen::VectorXd delta_x_;           // 15x1
    Eigen::MatrixXd P_;                 // 15x15 协方差矩阵

    // 传感器噪声参数（可配置）
    double noise_imu_gyro_;
    double noise_imu_acc_;
    double noise_odom_;
    double noise_lidar_;
    double noise_gps_;

    // IMU参数
    double freq_imu_;                   // IMU 频率
    double dt_imu_;                     // IMU 采样间隔
    const Eigen::Vector3d g_;           // 重力加速度（m/s2）
};

ESKF::ESKF() : g_(0, 0, 9.81) {
    // 1. 初始化绝对状态
    state_p_ = Eigen::Vector3d::Zero();
    state_v_ = Eigen::Vector3d::Zero();
    state_q_ = Eigen::Quaterniond::Identity();
    state_b_gyro_ = Eigen::Vector3d::Zero();
    state_b_acc_ = Eigen::Vector3d::Zero();

    // 2. 初始化误差状态
    delta_x_ = Eigen::VectorXd::Zero(15);
    P_ = Eigen::MatrixXd::Identity(15, 15) * 1e-6;
    P_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 1e-8;      // 姿态误差协方差
    P_.block<6, 6>(9, 9) = Eigen::MatrixXd::Identity(6, 6) * 1e-8;  // 零偏误差协方差

    // 3. 初始化默认噪声参数（可通过ROS节点重写）
    noise_imu_gyro_ = 0.001;    // rad/s
    noise_imu_acc_ = 0.01;      // m/s²
    noise_odom_ = 0.02;         // m
    noise_lidar_ = 0.05;        // m/rad
    noise_gps_ = 1.0;           // m

    // 4. IMU参数
    freq_imu_ = 125.0;
    dt_imu_ = 1.0 / freq_imu_;
}

// 配置传感器噪声参数（供ROS节点调用，实现松耦合）
void ESKF::set_noise_params(double imu_gyro_noise, double imu_acc_noise,
                        double odom_noise, double lidar_noise, double gps_noise) {
    noise_imu_gyro_ = imu_gyro_noise;
    noise_imu_acc_ = imu_acc_noise;
    noise_odom_ = odom_noise;
    noise_lidar_ = lidar_noise;
    noise_gps_ = gps_noise;
}

void ESKF::predict(const sensor_msgs::msg::Imu::SharedPtr& imu_msg) {
    // 1. 提取IMU数据并去零偏
    Eigen::Vector3d gyro(
        imu_msg->angular_velocity.x,
        imu_msg->angular_velocity.y,
        imu_msg->angular_velocity.z
    );
    Eigen::Vector3d acc(
        imu_msg->linear_acceleration.x,
        imu_msg->linear_acceleration.y,
        imu_msg->linear_acceleration.z
    );

    // 去除零偏
    gyro -= state_b_gyro_;
    acc -= state_b_acc_;

    // 2. 加速度去重力并转换到全局坐标系
    Eigen::Matrix3d R = state_q_.toRotationMatrix();
    Eigen::Vector3d acc_global = R * acc - g_;

    // 3. 离散积分更新绝对状态（简化版，实际可用IMU预积分库）
    // 姿态更新（小角度近似，四元数积分）
    Eigen::Quaterniond delta_q(1, gyro.x()*dt_imu_/2, gyro.y()*dt_imu_/2, gyro.z()*dt_imu_/2);
    state_q_ = (state_q_ * delta_q).normalized();
    // 速度更新
    state_v_ += acc_global * dt_imu_;
    // 位置更新
    state_p_ += state_v_ * dt_imu_ + 0.5 * acc_global * dt_imu_ * dt_imu_;

    // 4. 构建状态转移矩阵F（15x15，简化版）
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(15, 15);
    F.block<3, 3>(0, 6) = -R * skew(state_v_) * dt_imu_;  // δθ → δp
    F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity() * dt_imu_;  // δb_gyro → δθ

    // 5. 过程噪声协方差Q
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(15, 15);
    Q.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * pow(noise_imu_gyro_, 2) * dt_imu_;
    Q.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * pow(noise_imu_acc_, 2) * dt_imu_;

    // 6. 误差状态预测
    delta_x_ = F * delta_x_;
    // 7. 协方差预测
    P_ = F * P_ * F.transpose() + Q;
}

// 轮速记更新 (10Hz)
void ESKF::update_odom(const nav_msgs::msg::Odometry::SharedPtr& odom_msg) {
  // 1. 提取轮速计位置观测
  Eigen::Vector3d z_p(
    odom_msg->pose.pose.position.x,
    odom_msg->pose.pose.position.y,
    odom_msg->pose.pose.position.z
  );
  // 2. 观测模型：预测位置
  Eigen::Vector3d h_p = state_p_;
  // 3. 残差
  Eigen::VectorXd r(3);
  r << z_p - h_p;
  
  // 4. 观测矩阵H（3x15：仅观测位置δp）
  Eigen::MatrixXd H(3, 15);
  H.setZero();
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  // 5. 观测噪声协方差R
  Eigen::MatrixXd R = Eigen::Matrix3d::Identity() * pow(noise_odom_, 2);
  // 6. 卡尔曼增益
  Eigen::MatrixXd K = P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();
  // 7. 修正误差状态
  delta_x_ += K * r;
  // 8. 修正协方差
  P_ = (Eigen::MatrixXd::Identity(15, 15) - K * H) * P_;
  // 9. 误差反馈与重置
  feedback_and_reset();
}