#pragma once

#include <eigen3/Core>
#include <proj.h>

class GPS2UTM {
public:
    GPS2UTM() {
        // 初始化PROJ：WGS84（EPSG:4326）→ UTM Zone 50N（EPSG:32650，根据实际区域修改）
        proj_ctx_ = proj_context_create();
        proj_pj_ = proj_create_crs_to_crs(
            proj_ctx_,
            "EPSG:4326",   // 输入：WGS84经纬度
            "EPSG:32650",  // 输出：UTM 50N（可替换为对应区域的EPSG码）
            nullptr
        );

        if (!proj_pj_) {
            RCLCPP_ERROR(rclcpp::get_logger("GPS2UTM"), "PROJ initialization failed!");
        }
    }

    ~GPS2UTM() {
        if (proj_pj_) {
            proj_destroy(proj_pj_);
        }
        proj_context_destroy(proj_ctx_);
    }

    // GPS经纬度转UTM坐标（lat:纬度, lon:经度, alt:高度, utm:输出UTM坐标）
    bool convert(double lat, double lon, double alt, Eigen::Vector3d& utm) {
        if (!proj_pj_) {
            return false;
        }

        // PROJ坐标转换：lon/lat/alt → x/y/z
        PJ_COORD wgs84 = proj_coord(lon, lat, alt, wgs84);
        PJ_COORD utm_coord = proj_trans(proj_pj_, PJ_FWD, wgs84);

        utm << utm_coord.xy.x, utm_coord.xy.y, utm_coord.z;
        return true;
    }

private:
    PJ_CONTEXT* proj_ctx_;
    PJ* proj_pj_;
};