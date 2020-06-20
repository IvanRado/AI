# A simple implementation of user-user based collaboration
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# Load in the data
import os
if not os.path.exists('user2movie.json') or \
    not os.path.exists('movie2user.json') or \
    not os.path.exists('usermovie2rating.json') or \
    not os.path.exists('usermovie2rating_test.json'):
    import preprocess2dict

with open('user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
# Test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

if N > 10000:
    print("N =", N, "Are you sure you want to continue?")
    print("Comment out these lines if so...")
    exit()


# Calculating similarities
# Some configuration parameters
K = 25 # Number of neighbors we'd like to consider
limit = 5 # Number of common users must have in common in order to consider
neighbors = [] # Store neighbors in this list
averages = [] # Each user's average rating for later use
deviations = [] # Each user's deviation for later use

for i in range(N):
    # Find the 25 closest users to user i
    movies_i = user2movie[i]
    movies_i_set = set(movies_i)

    # Calculate the average and deviation
    ratings_i = { movie:usermovie2rating[(i, movie)] for movie in movies_i }
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = { movie:(rating - avg_i) for movie, rating in ratings_i.items() }
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    # Save these for later use
    averages.append(avg_i)
    deviations.append(dev_i)

    sl = SortedList()
    for j in range(N):

        if j != i:
            movies_j = user2movie[j]
            movies_j_set = set(movies_j)
            common_movies = (movies_i_set & movies_j_set)
            if len(common_movies) > limit:
                # Calculate the average and deviation
                ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j }
                avg_j = np.mean(list(j.values()))
                dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                # Calculate correlation coefficient
                numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
                w_ij = numerator / (sigma_i * sigma_j)

                # Insert into sorted list and truncate
                # Negate weight, because list is sorted ascending
                sl.add((-w_ij, j))
                if len(sl) > K:
                    del sl[-1]

    # Store the neighbors
    neighbors.append(sl)

    # Print out a useful indicator
    if i % i == 0:
        print(i)


# Using neighbors, calculate train and test MSE
def predict(i, m):
    # Calculate the weighted sum of deviations
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        # Remember the weight is stored as its negative
        # So the negative of the negative weight is the positive weight
        try:
            numerator += -neg_w * deviations[j][m]
            denominator += abs(neg_w)
        except KeyError:
            # Neighbor may not have rated the same moviem don't want to do dictionary lookup twice so just throw exception
            pass

    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]

    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction


train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
    # Calculate the prediction for this movie
    prediction = predict(i,m)

    # Save the prediction and target
    train_predictions.append(prediction)
    train_targets.append(target)

test_predictions = []
test_targets = []
# Same thing for the test set
for (i, m), target in usermovie2rating_test.items():
    # Calculate the prediction for this movie
    prediction = predict(i, m)

    # Save the prediction and target
    test_predictions.append(prediction)
    test_targets.append(target)

# Calculate the accuracy
def mse(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)

print("Train MSE:", mse(train_predictions, train_targets))
print("Test MSE:", mse(test_predictions, test_targets))