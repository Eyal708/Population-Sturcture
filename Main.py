from Migration import Migration
import numpy as np

M_test = np.array([[0, 0.3, 0.2], [0.1, 0, 0.4], [0.4, 0.2, 0]])
migration_mat = Migration(M_test)
A_test = migration_mat.produce_coefficient_matrix()
A_rank = np.linalg.matrix_rank(A_test)
print("Migration matrix is:\n", M_test, "\n")
print("The corresponding coefficients matrix is:\n", A_test, "\n")
print("The coefficients matrix rank is:", A_rank, "\n")
if A_rank == A_test.shape[0]:  # matrix is full rank
    T_test = migration_mat.produce_coalescence()
    print("The unique corresponding T matrix is:\n", T_test, "\n")
