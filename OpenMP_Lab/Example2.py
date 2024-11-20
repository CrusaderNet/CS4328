import numba as nb
import numpy as np
@nb.njit(parallel=True)
def parallel_matrix_multiplication(a, b):
    m, n = a.shape
    n, p = b.shape
    result = np.zeros((m, p))
    
    # Parallelize the matrix multiplication using OpenMP
    for i in nb.prange(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += a[i, k] * b[k, j]
    
    return result
# Create two matrices
a = np.random.rand(100, 100)
b = np.random.rand(100, 100)
# Execute the parallel matrix multiplication
result = parallel_matrix_multiplication(a, b)
print(result)
