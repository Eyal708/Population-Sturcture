import numpy as np
from Migration import Migration
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from Helper_funcs import compute_coalescence

# def change_mat(a):
#     a[0,0] = 0
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)
b = a[np.triu_indices(a.shape[0], 1)]
print(b)
# change_mat(a)
# print(a)
# x = fsolve(compute_coalescence, x0=np.array([1, 1, 1, 1, 1, 1]), args=([0.5, 0.4, 0.6], 3))
y = least_squares(compute_coalescence, x0=np.array([1, 1, 1, 1, 1, 1]), args=([0.5, 0.4, 0.6], 3), bounds=(0, np.inf)).x
# print(x)
print(y)

# c = a[b]
# print(a)
# print(b)
# print(c)
