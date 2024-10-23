from mpi4py import MPI
import numpy as np
import time

def matvec(comm, A_local, x_local):
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = len(x_local) * size  # Global vector size
    x_global = np.zeros(N)   # Allocate global vector

    comm.Allgather(x_local, x_global)  # Gather vector parts

    return np.dot(A_local, x_global)   # Compute local matrix-vector product

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 40  # Global matrix size (N x N)
assert N % size == 0, "Matrix size must be divisible by the number of processes."

local_n = N // size  # Local block size

A_local = np.random.random((local_n, N))  # Local matrix block
x_local = np.random.random(local_n)       # Local vector part

start_time = time.time()  # Start timer

y_local = matvec(comm, A_local, x_local)  # Perform matrix-vector product

end_time = time.time()  # Stop timer

# Gather results on the root process
y_global = None
if rank == 0:
    y_global = np.zeros(N)
comm.Gather(y_local, y_global, root=0)

# Print results on root process
if rank == 0:
    print("Matrix A (local):")
    print(A_local.reshape(local_n, N))
    print("\nVector x (local):")
    print(x_local)
    print("\nResult vector y (global):")
    print(y_global.reshape((N // size, size)))
    print(f"\nExecution time: {end_time - start_time:.10f} seconds")