# Demonstration of our confidence interval approximation wrt the beta posterior
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm

# Define trial # and probability for Bernoulli trial
T = 501
true_rate = 0.5
a, b = 1, 1
plot_indices = (10, 20, 30, 50, 100, 200, 500)
data = np.empty(T)

for i in range(T):
    x = 1 if np.random.random() < true_rate else 0 # "Flip coin" like action
    data[i] = x

    # update a and b
    a += x
    b += 1 - x

    if i in plot_indices:
        p = data[:i].mean() # Find the mean
        n = i + 1
        std = np.sqrt(p*(1-p)/n) 

        # Plot both distributions to compare
        x = np.linspace(0, 1, 200)
        g = norm.pdf(x, loc = p, scale = std)
        plt.plot (x, g, label = 'Gaussian Approximation')

        posterior = beta.pdf(x, a=a, b=b)
        plt.plot(x, posterior, label ='Beta Posterior')
        plt.legend()
        plt.title("N = %s" % n)
        plt.show()
