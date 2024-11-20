import random

def calc_pi(N):
    M = 0
    for i in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            M += 1
    return 4 * M / N

# Define the number of iterations
N = 1000000

# Calculate and print the approximation of pi
pi_approx = calc_pi(N)
print(f"Approximation of pi after {N} iterations: {pi_approx}")
