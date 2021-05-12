import numpy as np
from numba import guvectorize, int64

def add_2(a):
	return a + 2

@guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
def g(x, y, res):
    for i in range(x.shape[0]):
        res[i] = add_2(x[i])

a = np.arange(5)
print(a)
print(g(a, 2))
a = np.zeros((2, 4), dtype=int)
print(g(a, 2))
