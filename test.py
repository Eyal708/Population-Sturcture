import numpy as np
from Coalescence import Coalescence
from Fst import Fst
from Matrix_generator import generate_pseudo_random_fst_mat
from Helper_funcs import matrix_distance, diameter, check_constraint, \
    find_components, split_migration_matrix, split_migration
from Transformations import m_to_t, m_to_f



# wrong_cnt = 0
# bad_cnt = 0
# for i in range(500):
#     f = generate_pseudo_random_fst_mat(n=3)
#     F = Fst(f)
#     t = F.produce_coalescence(constraint=False)
#     T = Coalescence(t)
#     new_f = T.produce_fst()
#     if not(check_constraint(t)):
#         print(f"{t} is not a good t matrix!")
#         bad_cnt += 1
#     if not np.array_equal(np.round(f, 2), np.round(new_f, 2)):
#         print(f"Original F:\n{f}\n is no equal to new F:\n{new_f}")
#         wrong_cnt += 1

mat = np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
mat_2 = np.array([[0, 0, 0, 0, 0.1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0.1, 0, 0, 0], [0, 0, 0, 0, 0]])
mat_3 = np.array([[0, 0.1, 0, 0, 0], [0, 0, 0.1, 0, 0], [0, 0, 0, 0.1, 0], [0, 0, 0, 0, 0.1], [0, 0, 0, 0, 0]])
mat_4 = np.array([[0, 0, 0, 0.1], [0, 0, 0.1, 0], [0, 0.1, 0, 0], [0.1, 0, 0, 0]])
print(m_to_f(mat_4))
print(m_to_t(mat_4))
