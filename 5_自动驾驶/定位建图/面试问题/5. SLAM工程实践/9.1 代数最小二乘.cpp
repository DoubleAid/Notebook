#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

struct Plane {
    double a, b, c, d;
    double distanceToPoint(double x, double y, double z) const {
        return abs(a*x + b*y + c*z + d) / sqrt(a*a + b*b + c*c);
    }
};

Plane fitPlaneAlgebraic(const vector<Vector3d>& points) {
    int n = points.size();
    if (n < 3) {
        throw invalid_argument("至少需要3个点来拟合平面");
    }

    // 1. 计算点云中心
    Vector3d center(0, 0, 0);
    for (const auto& p : points) {
        center += p;
    }
    center /= n;

    // 2. 构建中心化点矩阵
    MatrixXd A(n, 3);
    for (int i = 0; i < n; ++i) {
        A.row(i) = points[i] - center;
    }

    // 3. 计算协方差矩阵 M = A^T * A
    Matrix3d M = A.transpose() * A;

    // 4. 特征值分解
    SelfAdjointEigenSolver<Matrix3d> eigensolver(M);
    if (eigensolver.info() != Success) {
        throw runtime_error("特征值分解失败");
    }

    // 5. 最小特征值对应的特征向量就是法向量
    Vector3d normal = eigensolver.eigenvectors().col(0);

    // 6. 计算d参数
    double d = -normal.dot(center);

    Plane plane;
    plane.a = normal(0);
    plane.b = normal(1);
    plane.c = normal(2);
    plane.d = d;
    
    return plane;
}