# An example of how we might use logistic regression for data that maps to concentric circles
import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

# Define the inner and outer radius for the dataset
R_inner = 5
R_outer = 10

# Create the inner circle
R1 = np.random.randn(int(N/2)) + R_inner
theta = 2*np.pi*np.random.random(int(N/2))
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

# Create the outer circle
R2 = np.random.randn(int(N/2)) + R_outer
theta = 2*np.pi*np.random.random(int(N/2))
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

# Create target vector
X = np.concatenate([ X_inner, X_outer])
T = np.array([0]*int((N/2)) + [1]*int((N/2)))

plt.scatter(X[:,0], X[:,1], c= T)
plt.show()

ones = np.ones((N,1))

r = np.zeros((N,1))

# Create a parametrized column to perform the feature engineering
for i in range(N):
    r[i] = np.sqrt(X[i,:].dot(X[i,:]))

Xb = np.concatenate((ones, r, X), axis = 1)

w = np.random.randn(D+2)

z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1+np.exp(-z))

Y = sigmoid(z)

def cross_entropy(Y, T):
    return -np.mean(T*np.log(Y) + (1-T)*np.log(1-Y))

# l1 regularized gradient descent
learning_rate = 0.0001
error = []
for i in range(5000):
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
