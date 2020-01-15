import numpy as np
import pandas as pd

def get_data():
    # Read in data and split into features and targets
    df = pd.read_csv("ecommerce_data.csv")
    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]

    # Normalize the data
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()


    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1

    Z = np.zeros((N,4))
    Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)

    return X2,Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2