#include <dphi_cloth_triangle_dX.h>

//compute 3x3 deformation gradient 
void dphi_cloth_triangle_dX(Eigen::Matrix3d &dphi, Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, Eigen::Ref<const Eigen::Vector3d> X) {

    dphi.setZero();

    Eigen::Vector3d X0 = V.row(element(0));
    Eigen::Vector3d X1 = V.row(element(1));
    Eigen::Vector3d X2 = V.row(element(2));
    Eigen::Matrix32d T;
    T << (X1 - X0), (X2 - X0);
    
    dphi.block<2,3>(1,0) = (T.transpose() * T).inverse() * T.transpose();
    dphi.block<1,3>(0,0) = -dphi.block<2,3>(1,0).colwise().sum();
}