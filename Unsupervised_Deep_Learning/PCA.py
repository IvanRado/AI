# Simple PCA implementation using numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import getKaggleMNIST

X_train, Y_train, X_test, Y_test = getKaggleMNIST()

# Decompose the covariance
covX = np.cov(X_train.T) 
lambdas, Q = np.linalg.eigh(covX)

# Arrange the lambdas in ascending order
idx = np.argsort(-lambdas)
lambdas = lambdas[idx]
lambdas = np.maximum(lambdas, 0)
Q = Q[:,idx]

# Perform PCA and plot the first two components
Z = X_train.dot(Q)
plt.scatter(Z[:,0], Z[:,1], s = 100, c=Y_train, alpha = 0.3)
plt.show()

# Plot the variances
plt.plot(lambdas)
plt.title("Variance of each component")
plt.show()

# Cumulative variance
plt.plot(np.cumsum(lambdas))
plt.title("Cumulative variance")
plt.show()