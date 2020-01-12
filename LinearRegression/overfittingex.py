# Create a deg degree polynomial and add the bias column
def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)]
    for d in range(deg):
        data.append(X**(d+1))
    return np.vstack(data).T

# Find the vector of weights
def fit(X,Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def fit_and_display(X, Y, sample, deg):
    # Split to train and test sets
    N = len(X)
    train_idx = np.random.choice(N, sample)
    X_train = X[train_idx]
    Y_train = Y[train_idx]

    plt.scatter(X_train, Y_train)
    plt.show()

    # Fit the polynomial
    X_train_poly = make_poly(X_train, deg)
    w = fit(X_train_poly, Y_train)

    # Display the polynomial
    X_poly = make_poly(X, deg)
    Yhat = X_poly.dot(w)
    plt.plot(X, Y)
    plt.plot(X, Yhat)
    plt.scatter(X_train, Y_train)
    plt.title("deg = %d" % deg)
    plt.show()



def get_mse(Y, Yhat):
    d = Y-Yhat
    return d.dot(d)/len(d)

def plot_train_test_curves(X, Y, sample=20, max_deg=20):
    N = len(X)
    train_idx = np.random.choice(N, sample)
    X_train = X[train_idx]
    Y_train = Y[train_idx]

    test_idx = [idx for idx in range(N) if idx not in train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    mse_trains = []
    mse_tests = []
    for deg in range(max_deg+1):
        X_train_poly = make_poly(X_train, deg)
        w = fit(X_train_poly, Y_train)
        Yhat_train = X_train_poly.dot(w)
        mse_train = get_mse(Y_train, Yhat_train)

        X_test_poly = make_poly(X_test, deg)
        Yhat_test = X_test_poly.dot(w)
        mse_test = get_mse(Y_test, Yhat_test)

        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

    plt.plot(mse_trains, label = "train mse")
    plt.plot(mse_tests, label = "test mse")
    plt.legend()
    plt.show()

    plt.plot(mse_trains, label = "train mse")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate a sine wave and plot 
    N = 100
    X = np.linspace(0, 6*np.pi, N)
    Y = np.sin(X)

    plt.plot(X,Y)
    plt.show()
    
    for deg in (5, 6, 7, 8, 9):
        fit_and_display(X, Y, 10, deg)
    plot_train_test_curves(X,Y)