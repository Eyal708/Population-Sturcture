import time

import numpy as np
import csv
from coalescence import Coalescence
from fst import Fst
from migration import Migration
from matrix_generator import generate_pseudo_random_fst_mat, generate_random_migration_mat
from helper_funcs import matrix_distance, diameter, check_constraint, \
    find_components, split_migration_matrix, split_migration, check_conservative
from utils import f_to_m, m_to_f

f = np.array([[0, 0.14, 0.14], [0.14, 0, 0.14], [0.14, 0.14, 0]])
#make a similar matrix to f but 5X5
f_2 = np.array([[0, 0.14, 0.14, 0.14, 0.14], [0.14, 0, 0.14, 0.14, 0.14], [0.14, 0.14, 0, 0.14, 0.14], [0.14, 0.14, 0.14, 0, 0.14], [0.14, 0.14, 0.14, 0.14, 0]])
F = Fst(f_2)
# vector of zeroes of size 9
# x0 = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
m, sol = F.produce_migration()
print(check_conservative(m))
print(m)
print(sol)
# basic_migration = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
# random_migration = generate_random_migration_mat(3)
# # print(m_to_f(random_migration))
# migration_matrix = generate_random_migration_mat(3)
# print("migration_matrix")
# print(basic_migration)
# m = Migration(basic_migration)
# print("m.produce_coefficient_matrix()")
# print(f"{m.produce_coefficient_matrix()}\n")
# print("m.coefficient_matrix_from_migration_wrapper()")
# print(m.coefficient_matrix_from_migration_wrapper())
# print("m.produce_coalescence()")
# load matrix.csv to matrix
# with open('matrix.csv', newline='') as csvfile:
#     data = list(csv.reader(csvfile))
#     print(data)
#     # convert data to numpy 2d array
#     matrix = np.array(data)
#     print(matrix)
#     # get time in seconds
#     start = time.time()
#     print(m_to_f(matrix))
#     end = time.time()
#     print(end - start)

# print(m.produce_coalescence())
# print("m.produce_coalescence_old()")
# print(m.produce_coalescence_old())
# print("fst matrix from produce_coalescence()")
# print(m_to_f(migration_matrix))
# print(f"fst matrix from produce_coalescence_old()\n{Coalescence(m.produce_coalescence_old()).produce_fst()}")
