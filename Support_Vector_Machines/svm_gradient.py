# A simple implementation of a SVM using projected gradient descent
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from util import get_spiral, get_xor, get_donut, get_clouds, plot_decision_boundary

import numpy as np
import matplotlib.pyplot as plt

# Kernels 
def linear(X1, X2, c=0):
    return X1.dot(X2.T) + c

def rbf(X1, X2, gamma=None):
    if gamma is None:
        gamma = 1.0 / X1.shape[-1]  # 1/D

    if np.ndim(X1) == 1 and np.ndim(X2) == 1:
        result = np.exp(-gamma * np.linalg.norm(X1 - X2)**2)
    elif (np.ndim(X1) > 1 and np.ndim(X2) == 1) or (np.ndim(X1) == 1 and np.ndim(X2) > 1):
        result = np.exp(-gamma * np.linalg.norm(X1 - X2, axis=1)**2)
    elif np.ndim(X1) > 1 and np.ndim(X2) > 1:
        result = np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis =2)**2)
    return result

def sigmoid(X1, X2, gamma=0.05, c=1):
    return np.tanh(gamma * X1.dot(X2.T) + c)

class SVM:
    def __init__(self, kernel, C=1.0):
        self.kernel = kernel
        self.C = C

    def _train_objective(self):
        return np.sum(self.alphas) - 0.5 * np.sum(self.YYK * np.outer(self.alphas, self.alphas))

    def fit(self, X, Y, lr=1e-5, n_iters=400):
        # We require these to make future predictions
        self.Xtrain = X
        self.Ytrain = Y
        self.N = X.shape[0]
        self.alphas = np.random.random(self.N)
        self.b = 0

        # Kernel matrix
        self.K = self.kernel(X,X)
        self.YY = np.outer(Y,Y)
        self.YYK = self.K * self.YY

        # Gradient ascent
        losses = []
        for _ in range(n_iters):
            loss = self._train_objective()
            losses.append(loss)
            grad = np.ones(self.N) - self.YYK.dot(self.alphas)
            self.alphas += lr * grad

            # Clip to meet our constraints
            self.alphas[self.alphas < 0] = 0
            self.alphas[self.alphas > self.C] = self.C

        # Distribution of biases
        idx = np.where((self.alphas) > 0 & (self.alphas < self.C))[0]
        bs = Y[idx] - (self.alphas * Y).dot(self.kernel(X, X[idx]))
        self.b = np.mean(bs)

        plt.plot(losses)
        plt.title("Loss per iteration")
        plt.show()

    def _decision_function(self, X):
        return (self.alphas * self.Ytrain).dot(self.kernel(self.Xtrain, X)) + self.b

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)

def medical():
    data = load_breast_cancer()
    X, Y = data.data, data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    return X_train, X_test, Y_train, Y_test, rbf, 1e-3, 200

def medical_sigmoid():
  data = load_breast_cancer()
  X, Y = data.data, data.target
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
  return X_train, X_test, Y_train, Y_test, sigmoid, 1e-3, 200

def xor():
  X, Y = get_xor()
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
  kernel = lambda X1, X2: rbf(X1, X2, gamma=5.)
  return X_train, X_test, Y_train, Y_test, kernel, 1e-2, 300

def donut():
  X, Y = get_donut()
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
  kernel = lambda X1, X2: rbf(X1, X2, gamma=5.)
  return X_train, X_test, Y_train, Y_test, kernel, 1e-2, 300

def spiral():
  X, Y = get_spiral()
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
  kernel = lambda X1, X2: rbf(X1, X2, gamma=5.)
  return X_train, X_test, Y_train, Y_test, kernel, 1e-2, 300

def clouds():
  X, Y = get_clouds()
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
  return X_train, X_test, Y_train, Y_test, linear, 1e-5, 400


if __name__ == '__main__':
  X_train, X_test, Y_train, Y_test, kernel, lr, n_iters = spiral()
  print("Possible labels:", set(Y_train))

  # make sure the targets are (-1, +1)
  Y_train[Y_train == 0] = -1
  Y_test[Y_test == 0] = -1

  # scale the data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # now we'll use our custom implementation
  model = SVM(kernel=kernel, C=1.0)

  t0 = datetime.now()
  model.fit(X_train, Y_train, lr=lr, n_iters=n_iters)
  print("train duration:", datetime.now() - t0)
  t0 = datetime.now()
  print("train score:", model.score(X_train, Y_train), "duration:", datetime.now() - t0)
  t0 = datetime.now()
  print("test score:", model.score(X_test, Y_test), "duration:", datetime.now() - t0)

  if X_train.shape[1] == 2:
    plot_decision_boundary(model)
