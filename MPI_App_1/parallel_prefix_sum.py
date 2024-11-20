#This program creates a random array of size 1000 and then computes the prefix sum of the data using the cumsum() function. The data is
#then divided among the processes using slicing, and each process computes the prefix sum of its local data. The local results are
#gathered on the root process using the gather() function, and then the prefix sum of the gathered data is computed on the root process
#using the cumsum() function.

from mpi4py import MPI 
import numpy as np

# Get the rank of the process and the size of the communicator
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# Create a random array of size 1000
data = np.random.rand(1000)

# Compute the prefix sum of the local data 
local_result = np.cumsum(data[rank::size])

# Gather the local results on the root process
results = MPI.COMM_WORLD.gather(local_result, root=0)

# On the root process, concatenate and compute the prefix sum of the gathered data

if rank == 0:
  results = np.concatenate(results) 
  prefix_sum = np.cumsum(results) 
  print(f"Prefix sum: {prefix_sum}")


