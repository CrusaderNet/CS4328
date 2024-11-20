import numba as nb
import numpy as np
@nb.njit(parallel=True)
def parallel_for_loop():
    n = 1000000
    result = np.zeros(n)
    
    # Parallelize the for loop using OpenMP
    for i in nb.prange(n):
        result[i] = i * i
    
    return result
# Execute the parallel for loop
result = parallel_for_loop()
print(result)
