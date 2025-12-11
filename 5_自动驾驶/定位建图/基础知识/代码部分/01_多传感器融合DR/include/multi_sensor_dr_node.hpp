#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "multi_sensor_dr/utils/thread_safe_buffer.hpp"
#include "multi_sensor_dr/utils/gps2utm.hpp"
#include "multi_sensor_dr/utils/lidar_registration.hpp"
#include "multi_sensor_dr/eskf/eskf.hpp"

class MultiSensorDRNode : public rclcpp::Node {
public:
    MultiSensorDRNode();
    ~MultiSensorDRNode() = default;

private:
    // 传感器回调函数
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);

    // 定时器回调（核心逻辑，125Hz）
    void timer_callback();

    // 发布融合后的数据（位姿、里程计、TF）
    void publish_fused_data(const rclcpp::Time& stamp);

    // 订阅者
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_gps_;

    // 发布者
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;

    // TF2广播器
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // 定时器
    rclcpp::TimerBase::SharedPtr timer_;

    // 数据缓存
    ThreadSafeBuffer<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
    ThreadSafeBuffer<nav_msgs::msg::Odometry::SharedPtr> odom_buffer_;
    ThreadSafeBuffer<sensor_msgs::msg::PointCloud2::SharedPtr> lidar_buffer_;
    ThreadSafeBuffer<sensor_msgs::msg::NavSatFix::SharedPtr> gps_buffer_;

    // 核心算法类（组合关系，松耦合）
    std::unique_ptr<ESKF> eskf_;
    std::unique_ptr<GPS2UTM> gps2utm_;
    std::unique_ptr<LidarRegistration> lidar_reg_;

    // 计数器与频率配置
    int imu_count_;
    int imu_per_odom_;
    int imu_per_lidar_;
    int imu_per_gps_;
};