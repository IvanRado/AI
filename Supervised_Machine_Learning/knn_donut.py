# Applying KNN to a donut like dataset
# Performs much better than linear regression without any modification

import matplotlib.pyplot as plt
from util import get_donut
from knn import KNN

if __name__ == "__main__":
    X, Y = get_donut()

    plt.scatter(X[:, 0], X[:, 1], s = 100, c =Y, alpha = 0.5)
    plt.show()

    knn = KNN(3)
    knn.fit(X, Y)
    print("Training accuracy:", knn.score(X, Y))

