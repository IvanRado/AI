import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

sms = pd.read_csv('spambase.data').values
np.random.shuffle(sms)

X = sms[:, :48]
Y = sms[:, -1]

X_train = X[:-100,]
Y_train = Y[:-100,]
X_test = X[-100:,]
Y_test = Y[-100:,]

model = MultinomialNB()
model.fit(X_train, Y_train)
print("Classification rate for NB:", model.score(X_test, Y_test))

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print("Classification rate for AdaBoost:", model.score(X_test, Y_test))