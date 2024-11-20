from numba import njit, prange
import numpy as np

@njit(parallel=True)
def parallel_matrix_multiplication(A, B):
    n, m = A.shape
    m, p = B.shape
    C = np.zeros((n, p))

    for i in prange(n):
        for j in prange(p):
            for k in prange(m):
                C[i, j] += A[i, k] * B[k, j]

    return C

# Example usage
A = np.random.rand(100, 100)
B = np.random.rand(100, 100)
C = parallel_matrix_multiplication(A, B)
print(C)
