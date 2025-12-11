// 工具类：激光配准

#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

class LidarRegistration {
public:
    LidarRegistration() {}
    ~LidarRegistration();

    // 激光帧间配准（prev:前一帧, curr:当前帧, trans:位置增量, rot:姿态增量）
    bool register_cloud(const sensor_msg::msg::PointCloud2::SharedPtr& prev_msg,
                        const sensor_msg::msg::PointCloud2::SharedPtr& curr_msg,
                        Eigen::Vector3d& trans,
                        Eigen::Quaterniond& rot);

private:
    // 可配置的配准参数（通过参数服务器传入，此处简化为常量）
    const float ndt_resolution_ = 1.0f;
    const float ndt_step_size_ = 0.1f;
    const float ndt_epsilon_ = 1e-6f;
    const int ndt_max_iter_ = 30;
    const float voxel_leaf_size_ = 0.5f;
};

// NDT 点云配准
bool LidarRegistration::register_cloud(const sensor_msg::msg::PointCloud2::SharedPtr& prev_msg,
                                        const sensor_msg::msg::PointCloud2::SharedPtr& curr_msg,
                                        Eigen::Vector3d& trans,
                                        Eigen::Quaterniond& rot) {
    // 1. 转换ROS点云到PCL点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr curr_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*prev_msg, *prev_cloud);
    pcl::fromROSMsg(*curr_msg, *curr_cloud);

    // 2. 下采样 (加速配准)
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_prev(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_curr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);

    vg.setInputCloud(prev_cloud);
    vg.filter(*filtered_prev);

    // 3. 初始化NDT配准
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setResolution(ndt_resolution_);
    ndt.setStepSize(ndt_step_size_);
    ndt.setTransformationEpsilon(ndt_epsilon_);
    ndt.setMaxIterations(ndt_max_iter_);

    // 4. 执行配准
    ndt.setInputSource(filtered_curr);
    ndt.setInputTarget(filtered_prev);
    pcl::PointCloud<pcl::PointXYZ> output_cloud;
    ndt.align(output_cloud);

    if (!ndt.hasConverged()) {
        return false;
    }

    // 5. 获取配准结果（变换矩阵）
    Eigen::Matrix4F trans_mat = ndt.getFinalTransformation();
    trans << trans_mat(0, 3) << trans_mat(1, 3) << trans_mat(2, 3);
    rot = Eigen::Quaterniond(trans_mat.block<3, 3>(0, 0).cast<double>());

    return true;
}