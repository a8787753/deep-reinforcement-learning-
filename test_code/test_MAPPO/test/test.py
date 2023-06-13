import numpy as np


a = np.random.random((2,2,1,3))

print(a)
print(a.shape)

b = np.concatenate(a)
print(b)
print(b.shape)
