# Implementation of knn of mnist dataset
import numpy as np
from sortedcontainers import SortedList
from util import get_data
from datetime import datetime

# Define KNN class
class KNN(object):
    def __init__(self, k):
        self.k = k

    # Simply store the data
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X): # Loop through the input features
            sl = SortedList()
            for j, xt in enumerate(self.X): # Find the distances between the current input and all others 
                diff = x - xt
                d = diff.dot(diff) # Square the difference
                if len(sl) < self.k: # If we have less than k points add the distance
                    sl.add((d, self.Y[j]))
                else:
                    if d < sl[-1][0]: # If the distance is smaller than the largest in the list, remove the old point and add the new one
                        del sl[-1]
                        sl.add((d, self.Y[j]))
            
            # Make a dictionary for the votes
            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items(): # Find the class with the most votes
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class # Add it as the prediction
        return y # Return the vector of predictions

    # Calculate the score for the prediction (accuracy)
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P==Y)

if __name__ == "__main__":
    X, Y = get_data(2000)
    Ntrain = 1000 # 50/50 train/test split
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    # Run for various values of k
    for k in (1,2,3,4,5):
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))

        t0 = datetime.now()
        print("Train accuracy:", knn.score(Xtrain, Ytrain))
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        t0 = datetime.now()
        print("Test accuracy:", knn.score(Xtest, Ytest))
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))