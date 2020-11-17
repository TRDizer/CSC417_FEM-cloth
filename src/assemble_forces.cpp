#include <assemble_forces.h>
#include <iostream>

void assemble_forces(Eigen::VectorXd &f, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::MatrixXd> qdot, Eigen::Ref<const Eigen::MatrixXd> dX,
                     Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> F, Eigen::Ref<const Eigen::VectorXd> a0,
                     double mu, double lambda) { 
        
    f.resize(q.size());
    f.setZero();
    
    for (int i = 0; i < F.rows(); i++) {
        Eigen::RowVector3i current_triangle = F.row(i);
        Eigen::Matrix<double, 1,9> tmp_row = dX.row(i);
        Eigen::Vector9d dV;
        dV.setZero();
        dV_membrane_corotational_dq(dV, q, Eigen::Map<const Eigen::Matrix3d>(tmp_row.data()), V, current_triangle, a0(i), mu, lambda);

        for (int vertex_i = 0; vertex_i < 3; vertex_i++) {
            f.segment<3>(current_triangle(vertex_i) * 3) -= dV.segment<3>(vertex_i * 3);
        }
    }
    
};
