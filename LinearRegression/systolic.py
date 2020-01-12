if __name__ == "__main__":
    # Example computing systolic blood pressure from age and weight
    # The data (x1, x2, x3) are for each patient

    # x1 is the systolic blood pressure
    # x2 is the age in years
    # x3 is the weight in pounds

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_excel('mlr02.xls')
    X = df.as_matrix()

    # Visualize x1 vs x2
    plt.scatter(X[:,1], X[:,0])
    plt.show()

    # Visualize x1 vs x3
    plt.scatter(X[:,2], X[:,0])
    plt.show()

    # Add bias term
    df['ones'] = 1
    Y = df['X1']
    X = df[['X2', 'X3', 'ones']]

    # Consider three cases: x2 only, x3 only, x2 and x3. Compare the R^2 values
    X2only = df[['X2', 'ones']]
    X3only = df[['X3', 'ones']]

    def calc_r2(X, Y):
        w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
        yhat = X.dot(w)

        ss_res = np.sum((Y-yhat)**2)
        ss_total = np.sum((Y-Y.mean())**2)
        r2 = 1 - ss_res/ss_total
        return r2

    print("R2 for x2 only:", calc_r2(X2only, Y))
    print("R2 for x3 only:", calc_r2(X3only, Y))
    print("R2 for x2 and x3:", calc_r2(X, Y))