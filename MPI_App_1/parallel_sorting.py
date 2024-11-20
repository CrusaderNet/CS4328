#A program that demonstrates how to use MPI to sort a large dataset in parallel

from mpi4py import MPI 
import numpy as np

# Get the rank of the process and the size of the communicator 
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# Create a random array of size 1000 
data = np.random.rand(1000)

# Sort the local data
local_data = np.sort(data[rank::size])

# Gather the local data on the root process
sorted_data = MPI.COMM_WORLD.gather(local_data, root=0)

# On the root process, concatenate and sort the gathered data 

if rank == 0:
  sorted_data = np.concatenate(sorted_data) 
  sorted_data = np.sort(sorted_data) 
  print(f"Sorted data: {sorted_data}")
