import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

def plot_decision_boundary(X, model):
  h = .02  # step size in the mesh
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))


  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, m_max]x[y_min, y_max].
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N,D)

sep = 2
X[:125] += np.array([sep, sep])
X[125:250] += np.array([sep, -sep])
X[250:375] += np.array([-sep, -sep])
X[375:] += np.array([-sep, sep])
Y = np.array([0]*125 + [1]*125 + [0]*125 + [1]*125 )

plt.scatter(X[:,0], X[:,1], s=100, c = Y, alpha = 0.5)
plt.show()

model = DecisionTreeClassifier()
model.fit(X, Y)
print("Score for 1 tree:", model.score(X,Y))

plt.scatter(X[:,0], X[:,1], s=100, c = Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()

class BaggedTreeClassifier():
    def __init__(self, B):
        self.B = B

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace = True)
            Xb = X[idx]
            Yb = Y[idx]

            model = DecisionTreeClassifier(max_depth=2)
            model.fit(Xb,Yb)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return np.round(predictions / self.B)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)

model = BaggedTreeClassifier(200)
model.fit(X,Y)

print("Score for bagged model:", model.score(X,Y))

plt.scatter(X[:,0], X[:,1], s=100, c = Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()