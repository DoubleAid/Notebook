#include <rclcpp/rclcpp.h>
#include "multi_sensor_dr/node/multi_sensor_dr_node.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc. argv);
    rclcpp::spin(std::shared_ptr<MultiSensorDRNode>());
    rclcpp::shutdown();
    return 0;
}