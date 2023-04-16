import numpy as np
from Coalescence import Coalescence
from Fst import Fst
from Transformation import m_to_f, m_to_t
from Helper_funcs import matrix_distance, diameter

a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
b = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
c = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3.3]])
print(matrix_distance(a, b))
print(matrix_distance(a, c))
print(matrix_distance(b, c))
print(diameter([a, b, c]))
