import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1+np.exp(-z))

# Generate a fat matrix with many useless features
N = 50
D = 50

X = (np.random.random((N,D)) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

# Set the target vector
T = np.round(sigmoid(X.dot(true_w)) + np.random.randn(N)*0.5)

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 30.0 # Higher punishment pulls all weights closer to 0

# Perform gradient descent with L1 regularization
for i in range(5000):
    Y = sigmoid(X.dot(w))
    delta = Y - T
    w -= learning_rate*(X.T.dot(delta) + l1*np.sign(w))

    cost = -np.mean(T*np.log(Y) + (1-T)*np.log(1-Y)) + l1*np.abs(w).mean()
    costs.append(cost)

plt.plot(costs)
plt.show()

plt.plot(true_w, label = 'true w')
plt.plot(w, label = 'map w')
plt.legend()
plt.show()
    