# Implementation of the Thompson sampling algorithm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        # p: The win rate
        self.p = p
        self.a = 1  # Initialize as random uniform
        self.b = 1  # Initialize as random uniform
        self.N = 0 

    def pull(self):
        # Draw a 1 with probability p
        return np.random.random() < self.p

    # Draws a sample from Beta(a,b)
    def sample(self):
        return beta.rvs(self.a, self.b)

    def update(self, X):
        self.a += X 
        self.b += 1 - X
        self.N += 1

def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label = f'real p: {b.p:.4f}, win rate = {b.a - 1}/{b.N}')
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        # Use Thompson Sampling to select the next bandit
        j = np.argmax([b.sample() for b in bandits])

        # Plot the posteriors
        if i in sample_points:
            plot(bandits, i)

        # Pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # Update the rewards log
        rewards[i] = x

        # Update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # To be plotted later
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    # Plot the results
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    # plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    experiment()
