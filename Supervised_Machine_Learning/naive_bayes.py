# Implementation of Naive Bayes on the MNIST dataset
import numpy as np
from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class NaiveBayes(object):
    def fit(self, X, Y, smoothing = 10e-3): # Characterize each target with a gaussian for the likelihood
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis = 0),
                'var': current_x.var(axis = 0) + smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y) # Prior is just proportion of occurences

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X, mean=mean, cov =var) + np.log(self.priors[c]) # Calculate probability of it being from each of the classes
        return np.argmax(P, axis = 1) # Return the highest probability option

if __name__ == "__main__":
    X, Y = get_data(10000)
    N_train = int(len(Y)/2)
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    model = NaiveBayes()
    t0 = datetime.now()
    model.fit(X_train, Y_train)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(X_train, Y_train))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))

    t0 = datetime.now()
    print("Test accuracy:", model.score(X_test, Y_test))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))




    
    

