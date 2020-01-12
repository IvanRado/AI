import numpy as np
import matplotlib.pyplot as plt

# Generate test data
N = 50

X = np.linspace(0, 10, 50)
Y = 0.5*X + np.random.randn(N)

# Create outliers
Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

# Add bias
X = np.vstack([np.ones(N), X]).T

# Calculate maximum likelihood case (no regularization)
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml)
plt.show()

# Calculate L2 case with lambda = 1000
l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml, label = 'Maximum Likelihood')
plt.plot(X[:,1], Yhat_map, label = 'Map')
plt.legend()
plt.show()
