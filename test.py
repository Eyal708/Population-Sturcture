import numpy as np

a = np.zeros((3, 5))
print(a.shape)
print(a)
print(np.sum([3 - i for i in range(1)]))
a[0, 2] = 3
a[1, 2] = 4
print(a)
print(np.sum(a[[2, 1], :]))
