#include <assemble_stiffness.h>
#include <iostream>
#include <vector>

void assemble_stiffness(Eigen::SparseMatrixd &K, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::VectorXd> qdot, Eigen::Ref<const Eigen::MatrixXd> dX,
                     Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> F, Eigen::Ref<const Eigen::VectorXd> a0, 
                     double mu, double lambda) { 
    K.resize(q.size(), q.size());   
    K.setZero();
    std::vector<Eigen::Triplet<double>> K_entries;

    for (int i = 0; i < F.rows(); i++){
        Eigen::RowVector3i current_triangle = F.row(i);
        Eigen::Matrix<double, 1,9> tmp_row = dX.row(i);
        Eigen::Matrix99d d2V;
        d2V_membrane_corotational_dq2(d2V, q, Eigen::Map<const Eigen::Matrix3d>(tmp_row.data()), V, current_triangle, a0(i), mu, lambda);

        // Iterate to populate 9 total d2V/d(corner_i)(corner_j) blocks
        for (int vertex_i = 0; vertex_i < 3; vertex_i++) {
            for (int vertex_j = 0; vertex_j < 3; vertex_j++) {
                // Iterate to populate 3x3 entries of each block
                for (int xyz_i = 0; xyz_i < 3; xyz_i++) {
                    for (int xyz_j = 0; xyz_j < 3; xyz_j++) {
                        K_entries.push_back(Eigen::Triplet<double>(
                            current_triangle(vertex_i) * 3 + xyz_i,
                            current_triangle(vertex_j) * 3 + xyz_j,
                            -d2V(vertex_i * 3 + xyz_i, vertex_j * 3 + xyz_j)
                        ));
                    }
                }
            }
        }
    }

    K.setFromTriplets(K_entries.begin(), K_entries.end());
};
