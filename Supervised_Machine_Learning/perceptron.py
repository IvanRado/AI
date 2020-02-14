import numpy as np
import matplotlib.pyplot as plt
from util import get_data as get_mnist
from datetime import datetime

def get_data():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300, 2))*2 - 1
    Y = np.sign(X.dot(w) + b)
    return X, Y

def get_simple_xor():
    X = np.array([0, 0], [0,1], [1,0], [1,1])
    Y = np.array([0, 1, 1, 0])
    return X, Y

class Perceptron:
    def fit(self, X, Y, learning_rate = 1.0, epochs = 1000):
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0

        N = len(Y)
        costs = []

        for epoch in range(epochs):
            Y_hat = self.predict(X)
            incorrect = np.nonzero(Y != Y_hat)[0]
            if len(incorrect) == 0:
                break
                
            i = np.random.choice(incorrect)
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            c = len(incorrect) / float(N)
            costs.append(c)
            
        print("Final w:", self.w, "Final b:", self.b, "Epochs:", (epoch + 1), "/", epochs)
        plt.plot(costs)
        plt.show()


    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

if __name__ == '__main__':
    X, Y = get_data()
    plt.scatter(X[:,0], X[:,1], c = Y, s = 100, alpha = 0.5)
    plt.show()

    Ntrain = len(Y) // 2
    X_train, Y_train = X[:Ntrain], Y[:Ntrain]
    X_test, Y_test = X[Ntrain:], Y[Ntrain:]

    # X, Y = get_mnist()
    # idx = np.logical_or(Y == 0, Y == 1)
    # X = X[idx]
    # Y = Y[idx]
    # Y[Y == 0] = -1

    model = Perceptron()
    t0 = datetime.now()
    model.fit(X_train, Y_train)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(X_train, Y_train))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))

    t0 = datetime.now()
    print("Test accuracy:", model.score(X_test, Y_test))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))


