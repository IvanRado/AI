import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process_data import get_binary_data

# Load the ecommerce data
X, Y = get_binary_data()
X, Y = shuffle(X, Y)

# Split into train and test sets
X_train = X[:-100]
Y_train = Y[:-100]
X_test = X[-100:]
Y_test = Y[-100:]

# Initialize weights
D = X.shape[1]
W = np.random.randn(D)
b = 0

# Define helper functions
def sigmoid(a):
    return 1/(1+np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))

train_costs = []
test_costs = []
learning_rate = 0.001

for i in range(10000):
    # Calculate predictions
    pY_train = forward(X_train, W, b)
    pY_test = forward(X_test, W, b)

    # Calculate cost 
    ctrain = cross_entropy(Y_train, pY_train)
    ctest = cross_entropy(Y_test, pY_test)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # Perform gradient descent
    W -= learning_rate*X_train.T.dot(pY_train - Y_train)
    b -= learning_rate*(pY_train - Y_train).sum()

    # Print cost to ensure it's decreasing
    if i % 1000 == 0:
        print(i, ctrain, ctest)

print("Final train classification rate:", classification_rate(Y_train, np.round(pY_train)))
print("Final test classification rate:", classification_rate(Y_test, np.round(pY_test)))

legend1, = plt.plot(train_costs, label = 'train cost')
legend2, = plt.plot(test_costs, label = 'test cost')
plt.legend([legend1, legend2])
plt.show()

