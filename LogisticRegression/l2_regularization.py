import numpy as np
import matplotlib.pyplot as plt

# Declare parameters and create test dataset
N = 100
D = 2

split = N//2

X = np.random.randn(N,D)

X[:split, :] = X[:split, :] + 2*np.ones((split, D))
X[split:, :] = X[split:, :] - 2*np.ones((split, D))
# Create the target vector
T = np.array([0]*50 + [1]*50)

# Add bias vector
ones = np.ones((N,1))
Xb = np.concatenate((X, ones), axis = 1)

# Randomly initialize weights
W = np.random.randn(D+1)

# Define helper functions
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def forward(X, W):
    return X.dot(W)

def cross_entropy_l2(Y, T, W, smoothing):
    return -np.mean(T*np.log(Y) + (1-T)*np.log(1-Y)) + (smoothing/2)*W.T.dot(W)

# Perform gradient descent
loss = []
smoothing_param = 0.1
learning_rate = 0.001
for i in range(1000):
    z = forward(Xb, W)
    Y = sigmoid(z)
    loss.append(cross_entropy_l2(Y,T,W,smoothing_param))
    if i % 100 == 0:    
        print("Loss:", loss[-1])
    
    W -=  learning_rate*(Xb.T.dot(Y-T) + smoothing_param*W)

print("Optimal W with L2 regularization:", W)