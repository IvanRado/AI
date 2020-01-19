# Sample of how we would apply AB Testing to the multi-armed bandit problem
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

# Define the bandits and number of trials
num_trials = 5000
bandit_probabilities = [0.2, 0.4, 0.7, 0.32, 0.4]

# Create the bandit (slot machine) class
class Bandit:
    def __init__(self, p): # Constructor
        self.p = p
        self.a = 1
        self.b = 1

    # Pull lever; get a result, "win" if the value is below p
    def pull(self):
        return np.random.random() < self.p

    # Sample from the beta distribution (our prior)
    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += x
        self.b += 1 - x

# Plot the distribution
def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label = 'real p: %.4f' % b.p)
    plt.title("Bandit distributions after {} trials".format(trial))
    plt.legend()
    plt.show()

# Run the experiment
def experiment():
    bandits = [Bandit(p) for p in bandit_probabilities]

    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 4500, 4999]

    for i in range(num_trials):
        bestb = None
        maxsample = -1
        allsamples = []
        for b in bandits: # Run for each bandit
            sample = b.sample() # Sample beta
            allsamples.append("%.4f" % sample)
            if sample > maxsample: # track the maxsample and best b value
                maxsample = sample
                bestb = b
            if i in sample_points:
                print("Current samples: {}".format(allsamples))
                plot(bandits, i)

            x = bestb.pull() # Play the machine
            bestb.update(x) # Update the a and b values

if __name__ == "__main__":
    experiment()
