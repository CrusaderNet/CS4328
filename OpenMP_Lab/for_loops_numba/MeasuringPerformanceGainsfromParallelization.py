import time
import numpy as np
from numba import njit, prange

# Define the array to sum
arr = np.random.rand(1000000)  # Array of 1,000,000 random numbers

# Without parallelization
def sum_array(arr):
    return np.sum(arr)

# With parallelization using Numba
@njit(parallel=True)
def parallel_sum_array(arr):
    total = 0.0
    for i in prange(len(arr)):
        total += arr[i]
    return total

# Measure execution time without parallelization
start_time = time.time()
sum_result = sum_array(arr)
end_time = time.time()
print("Non-parallel execution time:", end_time - start_time)
print("Sum (Non-parallel):", sum_result)

# Measure execution time with parallelization
start_time = time.time()
parallel_sum_result = parallel_sum_array(arr)
end_time = time.time()
print("Parallel execution time:", end_time - start_time)
print("Sum (Parallel):", parallel_sum_result)
