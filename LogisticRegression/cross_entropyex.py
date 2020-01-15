import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N,D)

# Create two separate outputs, centered at -2 and 2
X[:50, :] = X[:50, :] - 2*np.ones((50,D))
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

# Create target vector and add bias vector
T = np.array([0]*50 + [1]*50)

ones = np.ones((N,1))
Xb = np.concatenate((ones, X), axis = 1)

# Randomly initialize weights
w = np.random.randn(D + 1)

# Calculate model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1+ np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    err = 0
    for i in range(N):
        err -= (T[i]*np.log(Y[i]) + (1-T[i])*np.log(1-Y[i]))

    return err

print(cross_entropy(T,Y))

# Compare vs closed form solution

w = np.array([0,4,4])
z = Xb.dot(w)
Y = sigmoid(z)

print(cross_entropy(T,Y))

plt.scatter(X[:,0], X[:,1], c = T, s=100, alpha=0.5)

x_axis = np.linspace(-6,6,100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()