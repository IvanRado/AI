# Simple application of Support Vector Machines to MNIST
from sklearn.svm import SVC
from util import getKaggleMNIST
from datetime import datetime

# Get the data: https://www.kaggle.com/c/digit-recognizer
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

model = SVC(C=5., gamma=.05)

t0 = datetime.now()
model.fit(Xtrain, Ytrain)
print("train duration:", datetime.now() - t0)
t0 = datetime.now()
print("train score:", model.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
t0 = datetime.now()
print("test score:", model.score(Xtest, Ytest), "duration:", datetime.now() - t0)
