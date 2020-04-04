# Implementation of simple backward propagation for densely connected neural nets
import numpy as np
import matplotlib.pyplot as plt

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

# Derivative of hidden layer weights
def derivative_w2(Z, T, Y):
    return Z.T.dot(T - Y)

# Derivative of bias term for the hidden layer
def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)

# Derivative of the input layer weights
def derivative_w1(X, Z, T, Y, W2):
    return (T - Y).dot(W2.T) * Z * (1 - Z)

# Derivative of the input bias term
def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z).sum(axis=0))


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()

def main():
    # Generate 3 Gaussian clouds with distinct origins as the sample data
    Nclass = 500

    D = 2
    M = 3
    K = 3

    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, -2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass) 
    N = len(Y)

    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    # Randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    # Perform back propagation with gradient descent
    learning_rate = 10e-7
    costs = []
    for epoch in range(10000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print("Cost:", c, "Classification rate:", r)
            costs.append(c)

        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.show()



if __name__ == "__main__":
    main()