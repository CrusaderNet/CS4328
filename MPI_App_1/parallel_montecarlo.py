from mpi4py import MPI
import numpy as np

# Define the function to be integrated 
def f(x):
  return 4.0 / (1.0 + x**2)

# Get the rank of the process and the size of the communicator 
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# Set the number of samples and the integration limits 
n_samples = 100000
a = 0.0
b = 1.0

# Generate a random sample of points
x = np.random.uniform(a, b, n_samples // size)

# Compute the local integral
local_result = np.sum(f(x)) * (b - a) / n_samples

# Reduce the local integrals to a single value on the root process
result = MPI.COMM_WORLD.reduce(local_result, op=MPI.SUM, root=0)

# Print the result on the root processa
if rank == 0:
   print(f"Result: {result}")
