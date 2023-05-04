import numpy as np
from Coalescence import Coalescence
from Fst import Fst
from Matrix_generator import generate_pseudo_random_fst_mat
from Helper_funcs import matrix_distance, diameter

a = [[1, 2], [3, 4]
     ]
b = [[1, 2], [3, 4]
     ]
c = [[1, 2], [3, 4]]
print(a, "   ", b, "  ", c)
print(np.sum([a, b, c], axis=0))
