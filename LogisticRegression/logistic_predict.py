import numpy as np
from process_data import get_binary_data

X, Y = get_binary_data()

# Dimensionality
D = X.shape[1]
W = np.random.randn(D) # random weights
b = 0

# Activation function
def sigmoid(a):
    return 1/ (1+np.exp(-a))

# Making the prediction
def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X,W,b)
predictions = np.round(P_Y_given_X)

def classification_rate(Y, P):
    return np.mean(Y == P)

print("Score:", classification_rate(Y, predictions))