# Homework 4
# CS 4328
# Author: Seth Tourish

# Description: This code trains and tests 4 different machine learning models on the breast cancer dataset using MPI.
# In this code, I use broadcast to distribute the data to all processes and gather to collect the results at rank 0.
# We use 4 processes to train and test 4 different models in parallel, using a 80% training and 20% testing split.
# The models used are Logistic Regression, Decision Tree, Support Vector Machine (SVC), and Random Forest.
# The results are displayed in tabular format at rank 0, in order to provide a clear comparison of the models in a convenient format.

# Import necessary libraries
from mpi4py import MPI
import pandas as pd
import time
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Check for exactly 5 processes
if size != 4:
    if rank == 0:
        print("This code requires exactly 5 processes.")
    MPI.Finalize()
    exit()

# Load Data and Distribute data to processses
if rank == 0:
    data = pd.read_csv("Data11tumors.csv")
    x = data.drop(['Classes'], axis=1)
    y = data['Classes']
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)
else:
    xTrain = xTest = yTrain = yTest = None

# MPI Broadcast training and testing data to all processes
xTrain = comm.bcast(xTrain, root=0)
xTest = comm.bcast(xTest, root=0)
yTrain = comm.bcast(yTrain, root=0)
yTest = comm.bcast(yTest, root=0)

# Define models to be trained
models = [
    (LogisticRegression(), "Logistic Regression"),
    (DecisionTreeClassifier(), "Decision Tree"),
    (svm.SVC(kernel='rbf'), "SVC (RBF Kernel)"),
    (RandomForestClassifier(), "Random Forest")
]

# Training and testing all models on each rank
if rank < 4:

    # Select model based on rank
    model, model_name = models[rank]
    
    # Start timing
    start_time = time.time()
    start_cpu = os.times()

    # Train the model
    model.fit(xTrain, yTrain)

    # Stop timing
    end_time = time.time()
    end_cpu = os.times()

    # Predict and calculate accuracy
    y_pred = model.predict(xTest)
    accuracy = accuracy_score(yTest, y_pred) * 100

    # Calculate timing
    user_time = end_cpu.user - start_cpu.user
    wall_time = end_time - start_time

    # Collect results in a dictionary
    results = {
        "Model": model_name,
        "Accuracy (%)": f"{accuracy:.2f}",
        "User CPU Time (s)": f"{user_time:.4f}",
        "Wall Time (s)": f"{wall_time:.4f}"
    }

    # Gather all results at rank 0
    all_results = comm.gather(results, root=0)

    # Display results in tabular format at rank 0
    if rank == 0:
        print("\nCancer Classification Results:\n")
        print(tabulate(all_results, headers="keys", tablefmt="grid"))
MPI.Finalize()