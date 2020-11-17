#include <d2V_membrane_corotational_dq2.h>
#include <iostream>
#include <igl/svd3x3.h>
#include <Eigen/SVD>

void d2V_membrane_corotational_dq2(Eigen::Matrix99d &H, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::Matrix3d> dX, 
                          Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, double area, 
                          double mu, double lambda) {
    

    //SVD = USW^T
    Eigen::Matrix3d U;
    Eigen::Vector3d S; 
    Eigen::Matrix3d W; 
    Eigen::Matrix3d F; //deformation gradient
    
    double tol = 1e-5;
    
    //Compute SVD of F here
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

    F = x * X;
    // igl::svd3x3(F, U, S, W);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    S = svd.singularValues();
    W = svd.matrixV();
    
    //deal with singularity in the svd gradient
    if(std::fabs(S[0] - S[1]) < tol || std::fabs(S[1] - S[2]) < tol || std::fabs(S[0] - S[2]) < tol) {
        F += Eigen::Matrix3d::Random()*tol;
        Eigen::JacobiSVD<Eigen::Matrix3d> svd2(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd2.matrixU();
        W = svd2.matrixV();
        S = svd2.singularValues();
    }
    
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

    //TODO: compute H, the hessian of the corotational energy
    // Compute tensors for d2psi_dF2, dS, and define ds_ij
    Eigen::Matrix3d dS;
    dS.setZero();
    for (int i = 0; i < 3; i++) {
        dS(i,i) = mu * 2.0 * (S(i) - 1) + lambda * (S.sum() - 3);
    }

    // non-diagonal entries' mu vanishes...
    Eigen::Matrix3d d2S;
    d2S.setConstant(lambda);
    d2S.diagonal().array() += 2.0 * mu; 

    Eigen::Tensor3333d dU, dV;
    Eigen::Tensor333d dSigma; // dsigma_dF_ij
    dsvd(dU, dSigma, dV, F);


    // Hereby summon d2psi_dF!
    Eigen::Matrix99d d2psi;
    d2psi.setZero();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j ++) {
            Eigen::Vector3d ds_ij = d2S * dSigma[i][j];
            Eigen::Matrix3d d2psi_Fij =
                dU[i][j] * dS * W.transpose() + U * ds_ij.asDiagonal() * W.transpose() + U * dS * (dV[i][j]).transpose();
            
            // column major assembly
            // Eigen::Vector9d flat_d2psi_Fij = Eigen::Map<const Eigen::Vector9d>(d2psi_Fij.data(), d2psi_Fij.size());
            // d2psi.row(j * 3 + i) = flat_d2psi_Fij.transpose();

            // row major assembly
            d2psi_Fij.transposeInPlace();
            Eigen::Vector9d flat_d2psi_Fij = Eigen::Map<const Eigen::Vector9d>(d2psi_Fij.data(), d2psi_Fij.size());
            d2psi.row(i * 3 + j) = flat_d2psi_Fij.transpose();
        }
    }

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

    // row major dX
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
    double n_tilde_norm = (delta_x1.cross(delta_x2)).norm();

    auto get_cross_product_matrix = [](Eigen::Vector3d &v) {
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
        1.0 / n_tilde_norm * (Eigen::Matrix3d::Identity() - n * n.transpose()) * (get_cross_product_matrix(delta_x1) * cross_w_dx1 - get_cross_product_matrix(delta_x2) * cross_w_dx2);

    Eigen::Matrix99d dF = B + mat_N * nu;

    H = area * dF.transpose() * d2psi * dF;

    //fix errant eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix99d> es(H);
    
    Eigen::MatrixXd DiagEval = es.eigenvalues().real().asDiagonal();
    Eigen::MatrixXd Evec = es.eigenvectors().real();
    
    for (int i = 0; i < 9; ++i) {
        if (es.eigenvalues()[i]<1e-6) {
            DiagEval(i,i) = 1e-3;
        }
    }
    
    H = Evec * DiagEval * Evec.transpose();
    
}
