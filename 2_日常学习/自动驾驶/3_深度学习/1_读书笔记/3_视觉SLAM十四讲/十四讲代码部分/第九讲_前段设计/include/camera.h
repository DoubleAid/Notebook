#ifndef CAMERA_H
#define CAMERA_H

#include <memory> 

namespace slam {

class Camera {
public:
    typedef std::shared_ptr<Camera> Ptr;
    // fx_: 焦距在x方向上的像素单位表示
    // fy_: 焦距在y方向上的像素单位表示
    // cx_: 图像坐标系原点在像素坐标系中的x偏移量
    // cy_: 图像坐标系原点在像素坐标系中的y偏移量
    // depth_scale_: 深度图像中像素值与实际距离的比例因子
    float fx_, fy_, cx_, cy_, depth_scale_;

    Camera();
    Camera(float fx, float fy, float cx, float cy, float depth_scale=0):
        fx_(fx), fy_(fy), cx_(cx), cy_(cy), depth_scale_(depth_scale) {}
    Vector3d
};

}

#endif