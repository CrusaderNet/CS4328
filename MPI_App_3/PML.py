from mpi4py import MPI
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load the dataset
data = pd.read_csv("BDParkinson_Prediction.csv")
X = data.iloc[:, 0:4]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X,y, test_size=0.3, random_state=1)

# Split the dataset into chunks
chunk_size = len(X) // size
X_chunk = xTrain[rank * chunk_size: (rank + 1) * chunk_size]
y_chunk = yTrain[rank * chunk_size: (rank + 1) * chunk_size]

# Train a random forest classifier on the chunk
model = RandomForestClassifier()
model.fit(X_chunk, y_chunk)

# Evaluate the model on the test set
test_X = xTest
test_y = yTest
accuracy = model.score(test_X, test_y)

# Gather the accuracy scores from all processes
accuracies = comm.gather(accuracy, root=0)

# Print the final accuracy if this is the root process
if rank == 0:
    print(f"Final accuracy: {np.mean(accuracies)}")
