#include <dV_cloth_gravity_dq.h>

void dV_cloth_gravity_dq(Eigen::VectorXd &fg, Eigen::SparseMatrixd &M, Eigen::Ref<const Eigen::Vector3d> g) {

    fg.setZero();
    fg = M * -g.replicate(M.rows() / g.size(), 1);
    
}
