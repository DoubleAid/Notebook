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
    // 
 }