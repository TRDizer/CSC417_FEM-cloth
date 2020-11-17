#include <dV_membrane_corotational_dq.h>
#include <iostream>
#include <igl/svd3x3.h>
#include <Eigen/SVD>

void dV_membrane_corotational_dq(Eigen::Vector9d &dV, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::Matrix3d> dX, 
                          Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, double area, 
                          double mu, double lambda) {

    //Deformation Gradient
    Eigen::Matrix3d U;
    Eigen::Vector3d S; 
    Eigen::Matrix3d W; 

    //TODO: SVD Here
    Eigen::Vector3d X0 = V.row(element(0)).transpose();
    Eigen::Vector3d X1 = V.row(element(1)).transpose();
    Eigen::Vector3d X2 = V.row(element(2)).transpose();
    Eigen::Vector3d N = ((X1 - X0).cross(X2 - X0)).normalized();
    Eigen::Matrix43d X;
    X.block<3,3>(0,0) = dX;
    X.block<1,3>(3,0) = N.transpose(); // Transpose of unit normal for refernce coordinate 

    Eigen::Vector3d x0 = q.segment<3>(element(0) * 3);
    Eigen::Vector3d x1 = q.segment<3>(element(1) * 3);
    Eigen::Vector3d x2 = q.segment<3>(element(2) * 3);
    Eigen::Vector3d n = ((x1 - x0).cross(x2 - x0)).normalized();
    Eigen::Matrix34d x;
    x << x0, x1, x2, n; // (x0 x1 x2 n) where n is the unit normal for world coordinate

    Eigen::Matrix3d F = x * X;
    // igl::svd3x3(F, U, S, W);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    S = svd.singularValues();
    W = svd.matrixV();

    //Fix for inverted elements (thanks to Danny Kaufman)
    double det = S[0]*S[1];
    
     if(det <= -1e-10)
    {
        if(S[0] < 0) S[0] *= -1;
        if(S[1] < 0) S[1] *= -1;
        if(S[2] < 0) S[2] *= -1;
    }
    
    if(U.determinant() <= 0)
    {
        U(0, 2) *= -1;
        U(1, 2) *= -1;
        U(2, 2) *= -1;
    }
    
    if(W.determinant() <= 0)
    {
        W(0, 2) *= -1;
        W(1, 2) *= -1;
        W(2, 2) *= -1;
    }
    
    //TODO: energy model gradient 
    Eigen::Matrix3d dS;
    dS.setZero();
    for (int i = 0; i < 3; i++) {
        dS(i,i) = mu * 2.0 * (S(i) - 1) + lambda * (S.sum() - 3);
    }

    // Define dpsi_dF and flatten in column major fashion
    Eigen::Matrix3d dpsi = U * dS * W.transpose();
    // Extra line of code if one wants row major flattening. Reference: https://stackoverflow.com/a/22896750
    dpsi.transposeInPlace();
    Eigen::Vector9d flat_dpsi = Eigen::Map<const Eigen::Vector9d>(dpsi.data(), dpsi.size());

    // Define B
    Eigen::Matrix99d B;
    B.setZero();
    // column major
    //      dx/dX   D00        D10        D20        
    //      dy/dX       D00        D10        D20    
    //      dz/dX          D00        D10        D20 
    //      dx/dY   D01        D11        D21        
    // B =  dy/dY =     D01        D11        D21    
    //      dz/dY          D01        D11        D21 
    //      dx/dZ   D02        D12        D22        
    //      dy/dZ       D02        D12        D22    
    //      dz/dZ          D02        D12        D22 
    // for(int i = 0; i < 3; i++) {
    //     for(int j = 0; j < 3; j++) {
    //         B.block<3,3>(i * 3, j * 3) = Eigen::Matrix3d::Identity() * dX(j,i);
    //     }
    // }

    // row major
    //      dx/dX   D00        D10        D20        
    //      dx/dY   D01        D11        D21        
    //      dx/dZ   D02        D12        D22        
    //      dy/dX       D00        D10        D20    
    // B =  dy/dY =     D01        D11        D21    
    //      dy/dZ       D02        D12        D22    
    //      dz/dX          D00        D10        D20 
    //      dz/dY          D01        D11        D21 
    //      dz/dZ          D02        D12        D22 
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            B.block<3,1>(i * 3, j * 3 + i) = dX.row(j).transpose();
        }
    }

    // Define normal gradient
    Eigen::Matrix93d mat_N;
    mat_N.setZero();
    // column major
    //     Nx
    //        Nx
    //           Nx
    //     Ny
    // N =    Ny
    //           Ny
    //     Nz
    //        Nz
    //           Nz
    // for (int i = 0; i < 3; i++) {
    //     mat_N.block<3,3>(i * 3, 0) = Eigen::Matrix3d::Identity() * N(i);
    // }

    // row major
    //     Nx
    //     Ny
    //     Nz
    //        Nx
    // N =    Ny
    //        Nz
    //           Nx
    //           Ny
    //           Nz
    for (int i = 0; i < 3; i++) {
        mat_N.block<3,1>(i * 3, i) = N;
    }

    Eigen::Vector3d delta_x1 = x1 - x0;
    Eigen::Vector3d delta_x2 = x2 - x0;
    double n_tilde = (delta_x1.cross(delta_x2)).norm();

    auto get_cross_product_matrix = [](const Eigen::Vector3d &v) {
        Eigen::Matrix3d mat;
        mat << 0,       -v.z(),  v.y(),
               v.z(),   0,       -v.x(),
               -v.y(),  v.x(),   0;
        return mat;
    };

    Eigen::Matrix39d cross_w_dx1, cross_w_dx2;
    cross_w_dx1 << -Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity();
    cross_w_dx2 << -Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero();

    Eigen::Matrix39d nu = 
        1.0 / n_tilde * (Eigen::Matrix3d::Identity() - n * n.transpose()) * (get_cross_product_matrix(delta_x1) * cross_w_dx1 - get_cross_product_matrix(delta_x2) * cross_w_dx2);

    // Combine to form dF_dq and use it to complete dV
    // dF_dq = B + N * nu
    dV = area * (B + mat_N * nu).transpose() * flat_dpsi;
}
