#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <algorithm>

using namespace std;
using namespace Eigen;

struct Plane {
    double a, b, c, d;
    int inlier_count;
    
    double distanceToPoint(const Vector3d& p) const {
        return abs(a * p(0) + b * p(1) + c * p(2) + d) / sqrt(a*a + b*b + c*c);
    }
};

struct RANSACResult {
    Plane best_plane;
    vector<int> inlier_indices;
    int iterations;
};

// 检查三点是否共线
bool arePointsCollinear(const Vector3d& p1, const Vector3d& p2, const Vector3d& p3, double epsilon = 1e-6) {
    Vector3d v1 = p2 - p1;
    Vector3d v2 = p3 - p1;
    return v1.cross(v2).norm() < epsilon;
}

// 从三点拟合平面
Plane fitPlaneFrom3Points(const Vector3d& p1, const Vector3d& p2, const Vector3d& p3) {
    Vector3d v1 = p2 - p1;
    Vector3d v2 = p3 - p1;
    Vector3d normal = v1.cross(v2);
    
    if (normal.norm() < 1e-6) {
        throw invalid_argument("三点共线，无法确定平面");
    }
    
    normal.normalize();
    double d = -normal.dot(p1);
    
    return Plane{normal(0), normal(1), normal(2), d, 0};
}

// 用所有内点重新拟合平面（最小二乘）
Plane refinePlane(const vector<Vector3d>& points, const vector<int>& inlier_indices) {
    int n = inlier_indices.size();
    if (n < 3) {
        throw invalid_argument("内点数量不足");
    }
    
    // 提取内点
    vector<Vector3d> inliers;
    for (int idx : inlier_indices) {
        inliers.push_back(points[idx]);
    }
    
    // 使用最小二乘拟合
    Vector3d center(0, 0, 0);
    for (const auto& p : inliers) {
        center += p;
    }
    center /= n;
    
    MatrixXd A(n, 3);
    for (int i = 0; i < n; ++i) {
        A.row(i) = inliers[i] - center;
    }
    
    JacobiSVD<MatrixXd> svd(A, ComputeThinV);
    Vector3d normal = svd.matrixV().col(2);
    double d = -normal.dot(center);
    
    return Plane{normal(0), normal(1), normal(2), d, n};
}
