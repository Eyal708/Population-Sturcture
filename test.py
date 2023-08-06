import numpy as np
from Coalescence import Coalescence
from Fst import Fst
from Matrix_generator import generate_pseudo_random_fst_mat
from Helper_funcs import matrix_distance, diameter, check_constraint, \
    find_components, split_migration_matrix, split_migration
from Transformations import m_to_t, m_to_f, m_to_t_and_f

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

# mat = np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
# mat_2 = np.array([[0, 0, 0, 0, 0.1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0.1, 0, 0, 0], [0, 0, 0, 0, 0]])
# mat_3 = np.array([[0, 0.1, 0, 0, 0], [0, 0, 0.1, 0, 0], [0, 0, 0, 0.1, 0], [0, 0, 0, 0, 0.1], [0, 0, 0, 0, 0]])
# mat_4 = np.array([[0, 0, 0, 0.1], [0, 0, 0.1, 0], [0, 0.1, 0, 0], [0.1, 0, 0, 0]])
# mat_5 = np.array([[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
# mat_6 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
# mat_7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])
# mat_8 = np.array([[0, 0.5, 0, 0.5], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
# fst = m_to_f(mat_8)
# t = m_to_t(mat_8)
# result = m_to_t_and_f(mat_8)
# t_2, fst_2 = result[0], result[1]
# print(fst)
# print(fst_2)
# print(t)
# print(t_2)
# F = Fst(fst)
# new_t = F.produce_coalescence()
# print(new_t)
# a = np.array([1, 2, 3, 4])
# b = np.array([1, 1, 1, 1])
# print(np.linalg.multi_dot((a, a * a)))
# print(a + 2)
# print(a @ a)
# print(a[1:1])
# print(np.concatenate((a[:1], a[5:])))
# print(np.repeat(['x', 'm'], 100))
# print(np.tile(['x','m'],100))
# print(5 * [(0, 1)])
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
for i in range(m.shape[0]):
    print(np.sum(m[i, :]))
    print(np.sum(m[:, i]))
