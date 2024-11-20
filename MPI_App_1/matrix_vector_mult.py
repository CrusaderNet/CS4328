#This program creates a random 10x10 matrix and a random vector, and then divides the matrix into rows and distributes them among the
#processes using the array_split() function. Each process computes the dot product of its local rows and the vector using the
#dot()
#function,
#and then the results are gathered into a single array on the root process using the gather() function. Finally, the result is printed on the root
#process.

from mpi4py import MPI
import numpy as np

# Get the rank of the process and the size of the communicator
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# Create a random matrix and vector
matrix = np.ones((10, 10), dtype="uint8")*2
vector = np.ones(10, dtype="uint8")

# Divide the matrix into rows and distribute them among the processes
local_rows = np.array_split(matrix, size)[rank]
if rank == 1:
    print("LOCAL ROWS:")
    print(local_rows)

# Compute the local dot product
local_result = np.dot(local_rows, vector)
print(local_result)
# Gather the local results into a single array on the root process
result = MPI.COMM_WORLD.gather(local_result, root=0)

# Print the result on the root process
if rank == 0:
    print(matrix)
    print(vector)
    print(f"Result: {result}")
    combined_result = np.concatenate(result)  # Combine the list of arrays into one array
    print(f"Combined Result: {combined_result}")
