import numpy as np
from matplotlib import pyplot as plt
import random
import time
from numba import njit

def random_walker(n: int, length: int) -> np.ndarray:

    arr = np.zeros((n, length), dtype = float)

    for i in range(n):

        idx = 100

        for j in range(length):
            
            idx += idx*random.gauss(0.00015, 0.02)

            arr[i, j] = idx
    
    return arr

@njit
def nb_random_walker(n: int, length: int) -> np.ndarray:

    arr = np.zeros((n, length), dtype = float)

    for i in range(n):

        idx = 100

        for j in range(length):
            
            idx += idx*random.gauss(0.00015, 0.02)

            arr[i, j] = idx
    
    return arr


# --- monte carlo run without numba ------------------
starttime = time.time()
arr = random_walker(10000, 365)
endtime = time.time()

print("monte carlo random walk without NUMBA; 1st time: ")
print(endtime-starttime)

starttime = time.time()
arr = random_walker(10000, 365)
endtime = time.time()

print("monte carlo random walk without NUMBA; 2nd time: ")
print(endtime-starttime)

# --- monte carlo run with numba --------------------
starttime = time.time()
arr = nb_random_walker(10000, 365)
endtime = time.time()

print("monte carlo random walk with NUMBA; 1st time: ")
print(endtime-starttime)

starttime = time.time()
arr = nb_random_walker(10000, 365)
endtime = time.time()

print("monte carlo random walk with NUMBA; 2nd time: ")
print(endtime-starttime)

starttime = time.time()
arr = nb_random_walker(10000, 365)
endtime = time.time()

print("monte carlo random walk with NUMBA; 3rd time: ")
print(endtime-starttime)
