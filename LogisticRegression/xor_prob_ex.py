# Example showing the XOR problem with logistic regression
# If we have points in four corners of a plot, hard to draw a line to separate them
import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

T = np.array([0, 1, 1, 0])

ones = np.ones((N, 1))

# Create a new feature by multiplying x and y; renders them linearly separable
xy = np.matrix(X[:,0] * X[:,1]).T
Xb = np.array(np.concatenate((ones, xy, X), axis = 1))

w = np.random.randn(D+2)

z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1+np.exp(-z))

Y = sigmoid(z)

def cross_entropy(Y, T):
    return -np.mean(T*np.log(Y) + (1-T)*np.log(1-Y))

# l1 regularized gradient descent
learning_rate = 0.001
error = []
for i in range(15000):
    e = cross_entropy(Y, T)
    error.append(e)
    if i % 100 == 0:
        print(e)

    w += learning_rate * (np.dot((T-Y).T, Xb) - 0.01*w)

    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross-entropy")
plt.show()

print("Final w:", w)
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)