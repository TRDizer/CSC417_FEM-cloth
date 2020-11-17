#include <fixed_point_constraints.h>
#include <algorithm>
void fixed_point_constraints(Eigen::SparseMatrixd &P, unsigned int q_size, const std::vector<unsigned int> indices) {
    P.setZero();
    P.resize(q_size - indices.size() * 3, q_size);
    P.reserve(q_size - indices.size() * 3);

    int current_row = 0;
    for (int i = 0; i < q_size / 3; i++) {
        // if not in indices
        if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
            for (int j = 0; j < 3; j++) {
                P.insert(current_row, i * 3 + j) = 1;
                current_row++;
            }
        }
    }
}