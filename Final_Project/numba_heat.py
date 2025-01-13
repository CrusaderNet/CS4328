import numpy
from matplotlib import pyplot
import numba as nb
import time

# Numba function to solve the heat equation with parallelism
@nb.njit(parallel=True)
def solve_heat_equation_parallel(u, k, dt, dx, t_len, x_len):
    for t in range(1, t_len - 1):
        for x in nb.prange(1, x_len - 1):
            u[t + 1, x] = k * (dt / dx**2) * (u[t, x + 1] - 2 * u[t, x] + u[t, x - 1]) + u[t, x]
    return u

# Constants
length = 2
k = 0.466
temp_left = 200
temp_right = 200
total_time = 4
dx = 0.1
dt = 0.0001

# Create the x and t vectors
x_vec = numpy.linspace(0, length, int(length / dx))
t_vec = numpy.linspace(0, total_time, int(total_time / dt))

# Initialize the temperature array
u = numpy.zeros([len(t_vec), len(x_vec)])
u[:, 0] = temp_left
u[:, -1] = temp_right

# Measure execution time without Numba
start = time.time() # Start timer for no Numba
for t in range(1, len(t_vec) - 1):  # Loop through time
    for x in range(1, len(x_vec) - 1):  # Loop through space
        u[t + 1, x] = k * (dt / dx**2) * (u[t, x + 1] - 2 * u[t, x] + u[t, x - 1]) + u[t, x]    # Calculate temperature
end = time.time()   # End timer for no Numba
print(f"Execution Time Without Numba: {end - start:.4f} seconds")   # Print execution time without Numba

# Warm-up run for Numba
u_warmup = numpy.zeros([len(t_vec), len(x_vec)])    #initialize array for warm-up run
u_warmup[:, 0] = temp_left  #set dummy values for warm-up run
u_warmup[:, -1] = temp_right    #set dummy values for warm-up run
solve_heat_equation_parallel(u_warmup, k, dt, dx, len(t_vec), len(x_vec))  # Call to warm-up Numba function


# Reset u and measure execution time with Numba and parallelism
u = numpy.zeros([len(t_vec), len(x_vec)])
u[:, 0] = temp_left
u[:, -1] = temp_right


# Measure execution time with Numba and parallelism
start = time.time() # Start timer for Numba
u = solve_heat_equation_parallel(u, k, dt, dx, len(t_vec), len(x_vec))  # Call to solve the heat equation with Numba
end = time.time()   # End timer for Numba
print(f"Execution Time With Numba (Parallel): {end - start:.4f} seconds")

# Plot the final temperature distribution
pyplot.plot(x_vec, u[-1])
pyplot.ylabel("Temperature (C˚)")
pyplot.xlabel("Distance Along Rod (m)")
pyplot.title("Final Temperature Distribution")
pyplot.show()