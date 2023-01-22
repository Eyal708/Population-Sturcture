import numpy as np
def change_mat(a):
    a[0,0] = 0
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.triu_indices(a.shape[0])
change_mat(a)
print(a)

# c = a[b]
# print(a)
# print(b)
# print(c)
