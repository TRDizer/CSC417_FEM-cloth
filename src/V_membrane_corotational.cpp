#include <V_membrane_corotational.h>
#include <igl/svd3x3.h>
#include <Eigen/SVD>
#include <cmath>

//Allowed to use libigl SVD or Eigen SVD for this part
void V_membrane_corotational(double &energy, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::Matrix3d> dX, 
                          Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, double area, 
                          double mu, double lambda) {

    Eigen::Vector3d X0 = V.row(element(0)).transpose();
    Eigen::Vector3d X1 = V.row(element(1)).transpose();
    Eigen::Vector3d X2 = V.row(element(2)).transpose();
    Eigen::Matrix43d X;
    X.block<3,3>(0,0) = dX;
    X.block<1,3>(3,0) = ((X1 - X0).cross(X2 - X0)).normalized().transpose(); // Transpose of unit normal for refernce coordinate 

    Eigen::Vector3d x0 = q.segment<3>(element(0) * 3);
    Eigen::Vector3d x1 = q.segment<3>(element(1) * 3);
    Eigen::Vector3d x2 = q.segment<3>(element(2) * 3);
    Eigen::Matrix34d x;
    x << x0, x1, x2, ((x1 - x0).cross(x2 - x0)).normalized(); // (x0 x1 x2 n) where n is the unit normal for world coordinate

    Eigen::Matrix3d F = x * X;
    Eigen::Vector3d stretches;
    // igl::svd3x3(F, Eigen::Matrix3d(), stretches, Eigen::Matrix3d()); // We do not need the rotational terms
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    stretches = svd.singularValues();

    // psi = mu * [ sigma{i from 0 to 2} (si-1)^2 ] + lambda/2 * (s0 + s1 + s2 - 3)^2
    Eigen::Vector3d ones = Eigen::Vector3d::Ones();
    double psi = mu * (stretches - ones).dot(stretches - ones) + 0.5 * lambda * std::pow(stretches.sum() - 3, 2);

    // integration
    energy = area * psi; // h = 1
}
