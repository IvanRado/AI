# Implementation of epsilon greedy algorithm
import matplotlib.pyplot as plt
import numpy as np

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

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    options = [x for x in range(len(BANDIT_PROBABILITIES)) if x != optimal_j]
    print("Optimal j:", optimal_j)

    for i in range(NUM_TRIALS):

        # Use epsilon-greedy to select the next bandit
        if np.random.random() < EPS:
            num_times_explored += 1
            j = np.random.choice(options) 
        else:
            num_times_exploited += 1
            j = optimal_j 

        if j == optimal_j:
            num_optimal += 1

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
    print("# of times explored:", num_times_explored)
    print("# of times exploited:", num_times_exploited)
    print("# times selected optimal bandit:", num_optimal)

    # Plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == "__main__":
    experiment()
