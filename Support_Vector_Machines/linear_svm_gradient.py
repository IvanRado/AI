# An implementation of a linear SVM that uses gradient descent for training
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from util import get_clouds

# Define the Linear SVM using a class
class LinearSVM:
    def __init__(self, C = 1.0):
        self.C = C
    
    def _objective(self, margins):
        return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - margins).sum()

    def fit(self, X, Y, lr = 1e-5, n_iters=400):
        N, D = X.shape
        self.N = N
        self.w = np.random.randn(D)
        self.b = 0

        # Gradient descent
        losses = []
        for _ in range(n_iters):
            margins = Y * self._decision_function(X)
            loss = self._objective(margins)
            losses.append(loss)

            idx = np.where(margins < 1)[0]
            grad_w = self.w - self.C * Y[idx].dot(X[idx])
            self.w -= lr*grad_w
            grad_b = -self.C * Y[idx].sum()
            self.b -= lr * grad_b

        self.support_ = np.where((Y * self._decision_function(X)) <= 1)[0]
        print("num SVs:", len(self.support_))

        print("w:", self.w)
        print("b:", self.b)

        plt.plot(losses)
        plt.title("Loss per iteration")
        plt.show()

    def _decision_function(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)

def plot_decision_boundary(model, X, Y, resolution=100, colors=('b', 'k', 'r')):
    np.warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()

    # Generate a coordinate grid of shape [resolution x resolution]
    # Evaluate the model over the entire space
    x_range = np.linspace(X[:,0].min(), X[:,0].max(), resolution)
    y_range = np.linspace(X[:,1].min(), X[:,1].max(), resolution)
    grid = [[model._decision_function(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
    grid = np.array(grid).reshape(len(x_range), len(y_range))

    # Plot decision contours using grid and make a scatter plot of training data
    ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1,1,1), linestyles = ('--', '-', '--'), colors=colors)
    ax.scatter(X[:,0], X[:,1], c=Y, lw=0, alpha=0.3, cmap='seismic')

    # Plot support vectors (non-zero alphas) as circled points (linewidth > 0)
    mask = model.support_
    ax.scatter(X[:,0][mask], X[:,1][mask], c=Y[mask], cmap='seismic')

    # Debug
    ax.scatter([0], [0], c='black', marker='x')
    plt.show()

def clouds():
    X, Y = get_clouds()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)
    return X_train, X_test, Y_train, Y_test, 1e-3, 200

def medical():
    data = load_breast_cancer()
    X, Y = data.data, data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    return X_train, X_test, Y_train, Y_test, 1e-3, 200

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, lr, n_iters = clouds()
    print("Possible labels:", set(Y_train))

    # Make sure the targets are (-1, +1)
    Y_train[Y_train == 0] = -1
    Y_test[Y_test == 0] = -1

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearSVM(C=1.0)

    t0 = datetime.now()
    model.fit(X_train, Y_train, lr=lr, n_iters=n_iters)
    print("Train duration:", datetime.now() - t0)
    t0 = datetime.now()
    print("Train score:", model.score(X_train, Y_train), "Duration:", datetime.now() -t0)
    t0 = datetime.now()
    print("Test score:", model.score(X_test, Y_test), "Duration:", datetime.now() - t0)

    if X_train.shape[1] == 2:
        plot_decision_boundary(model, X_train, Y_train)

