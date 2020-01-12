if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    X = []
    Y = []
    
    # Read in the data
    for line in open("data_poly.csv"):
        x, y = line.split(',')
        X.append([1, float(x), float(x)*float(x)])
        Y.append(float(y))

    # Convert to arrays
    X = np.array(X)
    Y = np.array(Y)

    # Plot the linear relationship
    plt.scatter(X[:,1], Y)
    plt.show()

    # Calculate the weights
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    yhat = np.dot(X,w)

    # Plot the data and the line
    plt.scatter(X[:,1], Y)
    plt.plot(sorted(X[:,1]), sorted(yhat))
    plt.show()

    # Calculate the R^2
    ss_res = np.sum((Y - yhat)**2)
    ss_total = np.sum((Y - Y.mean())**2)
    R2 = 1 - ss_res/ss_total
    print("R^2 is:", R2)


    