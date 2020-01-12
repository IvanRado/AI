if __name__ == "__main__":
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate random test data
    x_test_data = np.linspace(0, 100, 51)
    y_test_data = np.random.randint(5, 100, 51)

    print(x_test_data)
    print(y_test_data)

    # Recall our linear regression is y_i = a*x_i + b
    # where a = (x_bar*y_bar - xy_bar)/(x^2_bar - x_bar^2)
    # and b = (x^2_bar * y_bar - x_bar*xy_bar)/(x^2_bar - x_bar^2)

    # First let's make our scatterplot
    plt.scatter(x_test_data, y_test_data)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title("Sample linear regression")

    # Now let's calculate our coefficients
    # Need x_bar, xy_bar, x^2_bar, x_bar^2, and y_bar
    x_bar = np.mean(x_test_data)
    y_bar = np.mean(y_test_data)
    
    x_sq_bar = np.mean(x_test_data**2)
    x_bar_sq = np.mean(x_test_data)**2

    xy_bar = np.mean(x_test_data * y_test_data)
    denominator =(x_sq_bar - x_bar_sq)

    # Find a and b
    a = (-x_bar*y_bar + xy_bar)/denominator
    print("Slope: {}".format(a))
    b = (x_sq_bar * y_bar - x_bar*xy_bar)/denominator
    print("Intercept: {}".format(b))

    # Define the line of best fit
    lobf = a*x_test_data + b

    plt.plot(x_test_data, lobf)
    plt.legend(["Line of best fit", "Data"])
    plt.show()

    # With a sample dataset, loaded from file
    X = []
    Y = []
    for line in open("data_1d.csv"):
        x, y = line.split(',')
        X.append(float(x))
        Y.append(float(y))

    X = np.array(X)
    Y = np.array(Y)

    plt.scatter(X, Y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title("Sample linear regression")

    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    
    x_sq_bar = np.mean(X**2)
    x_bar_sq = np.mean(X)**2

    xy_bar = np.mean(X * Y)
    denominator =(x_sq_bar - x_bar_sq)

    a = (-x_bar*y_bar + xy_bar)/denominator
    print("Slope: {}".format(a))
    b = (x_sq_bar * y_bar - x_bar*xy_bar)/denominator
    print("Intercept: {}".format(b))

    # Define the line of best fit
    lobf = a*X + b

    plt.plot(X, lobf)
    plt.legend(["Line of best fit", "Data"])
    plt.show()

    # Calculate the goodness of fit - R^2
    # Recall R^2 = 1 - (SS_res/SS_total)
    ss_res = np.sum((Y - lobf)**2)
    ss_total = np.sum((Y - Y.mean())**2)
    R_sq = 1 - (ss_res/ss_total)

    print("R^2: {}".format(R_sq))

