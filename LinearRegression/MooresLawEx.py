if __name__ == "__main__":
    import numpy as np
    import re
    import matplotlib.pyplot as plt

    X = []
    Y = []

    non_decimal = re.compile(r'[^\d]+')

    # Read in the data
    for line in open("moore.csv"):
        r = line.split('\t')
        x = int(non_decimal.sub('', r[2].split('[')[0]))
        y = int(non_decimal.sub('', r[1].split('[')[0]))
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    
    # Recall that Moore's law is exponential
    # This means its logarithm  is linear
    Y = np.log(Y)

    denom = X.dot(X) - X.mean()*X.sum()

    a = (X.dot(Y) - Y.mean() * X.sum())/denom
    b = (Y.mean()*X.dot(X) - X.mean() * X.dot(Y))/denom

    y_hat = a*X + b

    plt.scatter(X, Y)
    plt.plot(X, y_hat)
    plt.xlabel("Time")
    plt.ylabel("Log of transistor count")
    plt.title("Demonstration of Moore's Law")
    plt.show()

    ss_res = (Y - y_hat)**2
    ss_total = (Y - Y.mean())**2
    R2 = 1 - ss_res.sum()/ss_total.sum()
    print("a:", a, "b:", b)
    print("R^2 is:", R2)
    print("Time to double", np.log(2)/a, "years")
