import numpy as np
import matplotlib.pyplot as plt

# Generate test data; X is a fat matrix
N = 50
D = 50

X = (np.random.random((N,D)) - 0.5)*10
true_w = np.array([1, 0.5, -0.5] + [0]*(D-3)
Y = X.dot(true_w) + np.random.randn(N)*0.5

# Initialize w and l1 regularization parameters
costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 10.0 

for t in range(500):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w -learning_rate*(X.T.dot(delta) + l1*np.sign(w))

    mse = delta.dot(delta) / N
    costs.append(mse)

# Visualize cost function over iterations
plt.plot(costs)
plt.show()

print("Final w:", w)

# Plot true w and MAP w to see achieved sparsity
plt.plot(true_w, label = 'True w')
plt.plot(w, label = 'w_map')
plt.legend()
plt.show()