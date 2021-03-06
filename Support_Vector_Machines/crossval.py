# Simple application of SVMs with cross validation to medical diagnosis data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

# load the data
data = load_breast_cancer()

for C in (0.5, 1.0, 5.0, 10.0):
  pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=C))])
  scores = cross_val_score(pipeline, data.data, data.target, cv=5)
  print("C:", C, "mean:", scores.mean(), "std:", scores.std())
