# Simple application of SVMs to medical diagnosis 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load the data
data = load_breast_cancer()

# Split the data into train and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.33)

# Scale the data
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

model = SVC(kernel='rbf')
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))
