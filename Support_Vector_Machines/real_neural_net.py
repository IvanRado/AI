# Simple application of MLPClassifier to MNIST
import numpy as np

from util import getKaggleMNIST
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# get the data: https://www.kaggle.com/c/digit-recognizer
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

# scale first
pipeline = Pipeline([
  ('scaler', StandardScaler()),
  ('mlp', MLPClassifier(hidden_layer_sizes=(500,), activation='tanh')),
  # ('lr', LogisticRegression()),
])



t0 = datetime.now()
pipeline.fit(Xtrain, Ytrain)
print("train duration:", datetime.now() - t0)
t0 = datetime.now()
print("train score:", pipeline.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
t0 = datetime.now()
print("test score:", pipeline.score(Xtest, Ytest), "duration:", datetime.now() - t0)
