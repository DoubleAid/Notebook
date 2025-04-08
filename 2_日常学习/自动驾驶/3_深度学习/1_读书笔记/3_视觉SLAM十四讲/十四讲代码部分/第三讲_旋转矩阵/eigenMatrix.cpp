#include <iostream>
#include <ctime>

using namespace std;

// Eigen 部分
#include <Eigen/Core>
// 稠密矩阵的代数运算 （逆，特征值等）
#include <Eigen/Dense>

#define MATRIX_SIZE 50

/*********************************
 * 本程序演示了 Eigen 基本类型的使用
 * *******************************/

 int main(int argc, char** argv) {
    // Eigen 以矩阵为基本数据单元，他是一个模版类，他的前三个参数为：数据类型，行，列

    // 声明一个 2x3 的 float 矩阵
    Eigen::Matrix<float, 2, 3> matrix_23;
    // 同时，Eigen通过 typedef 定义了许多内置类型，不过底层仍是 Eigen::Matrix
    // 例如 Vector3d 实质上时 Eigen::Matrix<double, 3, 1>
    Eigen::Vector3d v_3d;
    // 还有 Matrix3d实质上时 Eigen::Matrix3d<double, 3, 3>
    Eigen::Matrix3d matrix_33 = Eigen::Matrix<double, 3, 3>
    // 如果不确定矩阵的大小，可以使用动态大小的矩阵
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    // 或者更简单一点
    Eigen::MatrixXd matrix_x;

    // 下面时对矩阵的操作
    // 输入数据
    matrix_23 << 1, 2, 3, 4, 5, 6;
    // 输出
    cout << matrix_23 << endl;

    // 使用（）访问矩阵中的元素
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 2; j++) {
         cout << matrix_23(i, j) << endl;
      }
    }

    v_3d << 3, 2, 1;

    // 矩阵和向量相乘，实际上是矩阵和矩阵相乘，但是不能混和两种不同类型的矩阵，这样写是错的
    // Eigen::Matrix<double, 2, 1> resutl_wrong_type = matrix_23 * v_3d;

    // 应该通过显式转换, 将 float 类型转化成 double 类型
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    
    // 
}