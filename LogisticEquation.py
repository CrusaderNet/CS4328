# Author: Seth Tourish
# CS4328
# This Program is a comparison of the Logistic Equation bifurcation diagram between Numba(Parallelized) and Non-Numba(Non-Parallelized) versions execution times

# Import the necessary libraries
import math
import random

import numpy as np 
import numba as nb
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

import statistics
from time import perf_counter

import warnings
warnings.filterwarnings("ignore")

# Logistic Map Function with Numba
@nb.jit(parallel=True)
def parallelLogistic (R, x0, N):    

    # Initialize arrays
    xSelect = np.empty(len(R) * 100)
    rSelect = np.empty(len(R) * 100)

    # Loop through the values of R with Numba parallelization
    for i in nb.prange(len(R)):
        x = x0
        # Loop through the values of N
        for j in range(N):
            x = R[i] * x * (1 - x)
            # Throw away the first 400 values
            if j >= 400:
                # Store the values of x and R
                xSelect[i * 100 + j - 400] = x
                rSelect[i * 100 + j - 400] = R[i]
    return xSelect, rSelect

# Logistic Map Function without Numba
def logis(r):
    x_list = [x0]    
    for i in range(N-1):
        x_list.append(r * x_list[-1] * (1 - x_list[-1]))
    return x_list[400:]

# Determine minimum precision for both times
def find_precision(time):
    formatted_time = f"{time:.10f}"  # Max precision
    return formatted_time.rstrip("0") if "0" in formatted_time else formatted_time

# Set up the variables
R_list = np.linspace(2.0, 4.0, 1000)
x0 = 0.3
N = 500

# Non Numba Version
x_select = []
R_select = []

non_numba_Start_Time = perf_counter()
for r in R_list:
    x_select.append(logis(r))
    R_select.append([r] * 100)
non_numba_End_Time = perf_counter()

# Convert to 1D array
x_select = np.array(x_select).ravel()
R_select = np.array(R_select).ravel()

# Calculate non-Numba execution time and print with precision
non_numba_execution_time = non_numba_End_Time - non_numba_Start_Time

# Numba Version
x_select_numba = []
R_select_numba_numba = []

# Numba Warmup
tempR, tempX = parallelLogistic(R_list, x0, N)

# Numba Execution with Time Tracking
numba_Start_Time = perf_counter()
x_select_numba, R_select_numba = parallelLogistic(R_list, x0, N)
numba_End_Time = perf_counter()

# Convert to 1D array
x_select_numba = np.array(x_select_numba).ravel()
R_select_numba = np.array(R_select_numba).ravel()

# Calculate non-Numba execution time and print with precision
non_numba_execution_time = non_numba_End_Time - non_numba_Start_Time

# Calculate Numba execution time and print with precision
numba_execution_time = numba_End_Time - numba_Start_Time

# Find Precise Execution Times
non_numba_precise = find_precision(non_numba_execution_time)
numba_precise = find_precision(numba_execution_time)

# Plot the bifurcation diagrams
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor='lightgray')

# Numba plot
axes[0].scatter(R_select_numba, x_select_numba, color='red', s=0.1)
axes[0].set_xlabel('The value of R')
axes[0].set_ylabel('The value of x')
axes[0].set_title(f'Numba Version\nBifurcation Diagram\n\n2.0 < R < 4.0  |  x0=0.3\n', fontsize=14)
numba_text = f"Numba time: {numba_precise} seconds"
axes[0].text(2.5, 0.9, numba_text, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

# Non-Numba plot
axes[1].scatter(R_select, x_select, color='blue', s=0.1)
axes[1].set_xlabel('The value of R')
axes[1].set_ylabel('The value of x')
axes[1].set_title(f'Non-Numba Version\nBifurcation Diagram\n\n2.0 < R < 4.0  |  x0=0.3\n', fontsize=14)
non_numba_text = f"Non-Numba time: {non_numba_precise} seconds"
axes[1].text(2.5, 0.9, non_numba_text, fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.5))

# Adjust layout and save
plt.tight_layout()
plt.savefig('bifurcation_diagrams_comparison.png')
plt.show()