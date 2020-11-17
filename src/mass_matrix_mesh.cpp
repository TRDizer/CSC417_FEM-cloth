#include <mass_matrix_mesh.h>
#include <vector>

void mass_matrix_mesh(Eigen::SparseMatrixd &M, Eigen::Ref<const Eigen::VectorXd> q, 
                         Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> F,
                         double density, Eigen::Ref<const Eigen::VectorXd> areas) {
    // Single triangle:
    //                       [ 1/6, 1/12, 1/12]
    // M_0 = h * rho * area  [1/12,  1/6, 1/12]
    //                       [1/12, 1/12,  1/6]
    // Assume h =1
    M.resize(q.size(), q.size());
    M.setZero();

    Eigen::Matrix3d M_0;
    M_0 << 1.0/6.0,  1.0/12.0, 1.0/12.0,
           1.0/12.0, 1.0/6.0,  1.0/12.0,
           1.0/12.0, 1.0/12.0, 1.0/6.0;
    M_0 *= density; // pre-multiply density since it is constant across the cloth

    std::vector<Eigen::Triplet<double>> M_entries;

    Eigen::RowVector3i current_triangle;
    for (int i = 0; i < F.rows(); i++) {
        current_triangle = F.row(i);

        for (int phi_i = 0; phi_i < 3; phi_i++) {
            for (int phi_j = 0; phi_j < 3; phi_j++) {
                for (int diag_i = 0; diag_i < 3; diag_i++) {
                    M_entries.push_back(
                        Eigen::Triplet<double>(
                            current_triangle(phi_i) * 3 + diag_i,
                            current_triangle(phi_j) * 3 + diag_i,
                            M_0(phi_i, phi_j) * areas(i)
                        )
                    );
                }
            }
        }
    }

    M.setFromTriplets(M_entries.begin(), M_entries.end());               
}
 