import numpy as np


a = np.random.random((3,3))

print(a)

print(np.all(a))

print(a==True)

a[0,0] = 0

print(a)

print(np.all(a))

print(a==True)

a[0,0] = True

print(a)

print(np.all(a))

print(a==True)

b = np.random.random((3,3))

print(b)

print(b[a==True])