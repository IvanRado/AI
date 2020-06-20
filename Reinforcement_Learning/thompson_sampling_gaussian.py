# Implementation of the Thompson sampling algorithm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

np.random.seed(1)
NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [1, 2, 3]

class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        # Parameters for mu - prior is N(0,1)
        self.predicted_mean = 0
        self.lambda_ = 0  
        self.sum_x = 0  # For convenience
        self.tau = 1  
        self.N = 0 

    def pull(self):
        # Play the arm
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    # Draws a sample from Beta(a,b)
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.true_mean

    def update(self, X):
        self.lambda_ += self.tau
        self.sum_x += X
        self.predicted_mean = self.tau*self.sum_x / self.lambda_
        self.N += 1
        

def plot(bandits, trial):
    x = np.linspace(-3, 6, 200)
    for b in bandits:
        y = norm.pdf(x, b.predicted_mean, np.sqrt(1 / (b.lambda_ + 0.00001)) )
        plt.plot(x, y, label = f'real mu: {b.true_mean:.4f}, times played: {b.N}')
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
