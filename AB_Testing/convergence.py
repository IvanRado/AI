# An example of how Thompson sampling converges to the true mean
import matplotlib.pyplot as plt
import numpy as np
from bayesian_bandit import Bandit # Make use of our bandit class

def run_experiment(p1, p2, p3, N):
    bandits = [Bandit(p1), Bandit(p2), Bandit(p3)]

    data = np.empty(N)

    for i in range(N):
        j = np.argmax([b.sample() for b in bandits]) # Select the best option
        x = bandits[j].pull() # Pull this lever
        bandits[j].update(x) # Update the distribution

        data[i] = x # Store the data
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1) # Find the average

    plt.plot(cumulative_average) # Plotting the average of the decision
    plt.plot(np.ones(N)*p1) # Plotting of each bandit's true probability
    plt.plot(np.ones(N)*p2)
    plt.plot(np.ones(N)*p3)
    plt.ylim((0,1))
    plt.xscale('log') # Easier to visualize due to # of iterations
    plt.show()

run_experiment(0.2, 0.25, 0.3, 100000)
