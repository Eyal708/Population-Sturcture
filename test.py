import numpy as np

a = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
x = a.argmin(axis=1)
y = np.diag_indices(3)[0]
print(not np.any(x != y))
