# generate_data.py

import numpy as np
np.random.seed(0)

# Generate 10 arrays of shape (5000, 10) and save them as .npy files
for i in range(10):
  data = np.random.randn(5000, 10)
  np.save(f'data_{i}.npy', data)
