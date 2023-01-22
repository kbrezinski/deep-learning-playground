
import numpy as np

# shape (3, 1)
x_vec, w_vec = np.array([1., 2., 3.]), np.array([0.1, 0.3, 0.5])

print(x_vec.shape, w_vec.shape)

# (3, 1) .dot (3.1); works like inner product
print(x_vec.dot(w_vec))

