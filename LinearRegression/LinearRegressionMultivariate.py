if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    X = []
    Y = []

    # Read in the data
    for line in open('data_2d.csv'):
        x1, x2, y = line.split(',')
        X.append([float(x1),float(x2),1])
        Y.append(float(y))

    # Convert to NumPy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X[:,0], X[:,1], Y)
    plt.show()

    # Calculate weights of the model
    # Alternative method: w = np.matmul((np.linalg.inv(np.matmul(X.T, X))), np.matmul(X.T, Y))
    # This is our eqn
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    yhat = np.dot(X, w)
    
    # Calculate R2
    ss_res = np.sum((Y-yhat)**2)
    ss_total = np.sum((Y - Y.mean())**2)
    R2 = 1 - ss_res/ss_total
    print(R2)
