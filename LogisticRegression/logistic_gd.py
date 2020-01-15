import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

# Create two Gaussian clouds
X = np.random.randn(N,D)
X[:50, :] = X[:50, :] - 2*np.ones((50,D))
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

# Add bias term
ones = np.ones((N,1))
Xb = np.concatenate((ones, X), axis = 1)
# Target vector
T = np.array([1]*50 + [0]*50)

# Randomly initialize weights
w = np.random.randn(D+1)

print("w:", w)

# Define sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cross entropy loss
def cross_entropy(T, Y):
    err = 0
    for i in range(len(Y)):
        if T[i] == 1:
            err -= np.log(Y[i])
        else:
            err -= np.log(1-Y[i])

    return err

# Set up gradient descent
learning_rate = 0.1
iterations = 100
cross_entropy_loss = []

# Perform gradient descent
for i in range(iterations):
    z = Xb.dot(w)
    Y = sigmoid(z)
    d = Xb.T.dot((Y-T))
    cross_entropy_loss.append(cross_entropy(T,Y))
    if i%10 == 0:
        print(cross_entropy_loss[-1])
    w = w - learning_rate*d

print("Optimal weights after 100 iterations:", w)

plt.plot(cross_entropy_loss)
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy Loss")
plt.title("Cross Entropy Loss during Gradient Descent")
plt.show()

