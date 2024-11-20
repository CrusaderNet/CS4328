import numpy as np
from numba import njit, prange

@njit(parallel=True)
def sum_of_squares(n):
    result = 0
    for i in prange(n):
        result += i ** 2
    return result

n = 1000000
print(sum_of_squares(n))
