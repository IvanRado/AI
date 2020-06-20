# Implementation of UCB (upper confidence bound) algorithm
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, log

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        # p: The win rate
        self.p = p
        self.p_estimate = 0 
        self.N = 0 

    def pull(self):
        # Draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, X):
        self.N = self.N + 1
        self.p_estimate = self.p_estimate + (1/self.N) * (X - self.p_estimate) 

def ucb(mean, n, nj):
    return mean + np.sqrt(2 * np.log(n) / nj)

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    total_plays = 0

    # Initialization: play each bandit once
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

    for i in range(NUM_TRIALS):
        # Use UCB to select the next bandit
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])

        # Pull the arm for the bandit with the largest sample
        x = bandits[j].pull()
        total_plays += 1

        # Update the rewards log
        rewards[i] = x

        # Update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # To be plotted later
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    # Plot the results
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    experiment()
