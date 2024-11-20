from numba import njit, prange

@njit(parallel=True)
def parallel_sum_array(arr):
    total = 0
    for i in prange(len(arr)):
        total += arr[i]
    return total

# Example usage
import numpy as np
arr = np.arange(1000000)
print(parallel_sum_array(arr))
