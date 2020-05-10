# Simple use case for an RBF network
from sklearn.svm import SVC
from util import getKaggleMNIST
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem

# Get the data: https://www.kaggle.com/c/digit-recognizer
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

# Multiple Nystroem
n_components = 1000
featurizer = FeatureUnion([
  ("rbf0", Nystroem(gamma=0.05, n_components=n_components)),
  ("rbf1", Nystroem(gamma=0.01, n_components=n_components)),
  ("rbf2", Nystroem(gamma=0.005, n_components=n_components)),
  ("rbf3", Nystroem(gamma=0.001, n_components=n_components)),
  ])
pipeline = Pipeline([('rbf', featurizer), ('linear', SGDClassifier(max_iter=1e6, tol=1e-5))])


t0 = datetime.now()
pipeline.fit(Xtrain, Ytrain)
print("train duration:", datetime.now() - t0)
t0 = datetime.now()
print("train score:", pipeline.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
t0 = datetime.now()
print("test score:", pipeline.score(Xtest, Ytest), "duration:", datetime.now() - t0)
