# Implementation of optimistic initial values algorithm
import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        # p: The win rate
        self.p = p
        self.p_estimate = 10
        self.N = 1. 

    def pull(self):
        # Draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, X):
        self.N = self.N + 1
        self.p_estimate = self.p_estimate + (1/self.N) * (X - self.p_estimate) 

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):

        # Select the best option
        j = np.argmax([b.p_estimate for b in bandits])

        # Pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # Update the rewards log
        rewards[i] = x

        # Update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # Print mean estimates for each bandit
    for b in bandits:
        print("Mean estimate:", b.p_estimate)

    # Print total reward
    print("Total reward earned:", rewards.sum())
    print("Overall win rate:", rewards.sum() / NUM_TRIALS)

    # Plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == "__main__":
    experiment()
