# Implementation of simple forward propagation for densely connected neural nets
import numpy as np
import matplotlib.pyplot as plt

# Generate 3 Gaussian clouds with distinct origins as the sample data
Nclass = 500

X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, -2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
plt.scatter(X[:,0], X[:,1], c=Y, s = 100, alpha=0.5)
plt.show()

# Two inputs into two layers with three neurons
D = 2
M = 3
K = 3

# Randomly initialize weights
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

# Forward pass with sigmoid activation in the first layer and
# softmax activation in the second
def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

# Calculate classification rate for the randomly initialized weights
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total +=1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

assert(len(P) == len(Y))

print("Classification rate for randomly chosen weights:{}".format(classification_rate(Y, P)))

        