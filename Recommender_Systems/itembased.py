# A simple implementation of item based collaborative filtering
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
# The test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

if M > 2000:
  print("N =", N, "Are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()

K = 20 # Number of neighbors we'd like to consider
limit = 5 # Number of common movies users must have in common in order to consider
neighbors = [] # Store neighbors in this list
averages = [] # Each item's average rating for later use
deviations = [] # Each item's deviation for later use

for i in range(M):
  # Find the K closest items to item i
  users_i = movie2user[i]
  users_i_set = set(users_i)

  # Calculate avg and deviation
  ratings_i = { user:usermovie2rating[(user, i)] for user in users_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { user:(rating - avg_i) for user, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # Save these for later use
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(M):
    # Don't include yourself
    if j != i:
      users_j = movie2user[j]
      users_j_set = set(users_j)
      common_users = (users_i_set & users_j_set) # intersection
      if len(common_users) > limit:
        # Calculate avg and deviation
        ratings_j = { user:usermovie2rating[(user, j)] for user in users_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { user:(rating - avg_j) for user, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # Calculate correlation coefficient
        numerator = sum(dev_i[m]*dev_j[m] for m in common_users)
        w_ij = numerator / (sigma_i * sigma_j)

        # Insert into sorted list and truncate negate weight, because list is sorted ascending maximum value (1) is "closest"
        sl.add((-w_ij, j))
        if len(sl) > K:
          del sl[-1]

  # Store the neighbors
  neighbors.append(sl)

  # Print out useful things
  if i % 1 == 0:
    print(i)



# Using neighbors, calculate train and test MSE
def predict(i, u):
  # Calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # Remember, the weight is stored as its negative
    # So the negative of the negative weight is the positive weight
    try:
      numerator += -neg_w * deviations[j][u]
      denominator += abs(neg_w)
    except KeyError:
      # Neighbor may not have been rated by the same user don't want to do dictionary lookup twice so just throw exception
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]

  prediction = min(5, prediction)
  prediction = max(0.5, prediction) # Min rating is 0.5
  return prediction

train_predictions = []
train_targets = []
for (u, m), target in usermovie2rating.items():
  # Calculate the prediction for this movie
  prediction = predict(m, u)

  # Save the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []
# Same thing for test set
for (u, m), target in usermovie2rating_test.items():
  # Calculate the prediction for this movie
  prediction = predict(m, u)

  # Save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)


# Calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('Train MSE:', mse(train_predictions, train_targets))
print('Test MSE:', mse(test_predictions, test_targets))