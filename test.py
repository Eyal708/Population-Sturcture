import numpy as np

x = np.array([1, 2, 3, 4])
x = x.reshape(x.shape[0], 1)
print(x.shape)
x_t = np.transpose(x)
print(np.matmul([x, x_t]))

