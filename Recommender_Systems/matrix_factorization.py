# Simple implementation of matrix factorization using alternating least squares
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

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

# Config parameters
K = 10  # Latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

# Grab loss
def get_loss(d):
    # d: (user_id, movie_d) -> rating
    N = float(len(d))
    sse = 0
    for k, r in d.items():
        i, j = k
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += (p - r)*(p - r)
    return sse / N

# Training parameters
epochs = 25
reg = 20.  # Regularization penalty
train_losses = []
test_losses = []
for epoch in  range(epochs):
    print("epoch:", epoch)
    epoch_start = datetime.now()

    # Perform updates
    t0 = datetime.now()
    for i in range(N):
        # Update W
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # Update b
        bi = 0
        for j in user2movie[i]:
            r = usermovie2rating[(i,j)]
            matrix += np.outer(U[j], U[j])
            vector += (r - b[i] - c[j] - mu)*U[j]

        # Set the updates
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(user2movie[i]) + reg)

    if i % (N//10) == 0:
        print("i:", i, "N:", N)
    print("Updated W and b:", datetime.now() - t0)

    # Update U and c
    t0 = datetime.now()
    for j in range(M):
        # for U
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # for c
        cj = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i,j)]
                matrix += np.outer(W[i], W[i])
                vector += (r - b[i] - c[j] - mu)*W[i]
                cj += (r - W[i].dot(U[j]) - b[i] - mu)

            # Apply updates
            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(movie2user[j]) + reg)

            if j % (M//10) == 0:
                print("j:", j, "M:", M)
        except KeyError:
            # Possible we have no ratings for a movie
            pass
    
    print("Updated U  and c:", datetime.now() - t0)
    print("epoch duration:", datetime.now() - epoch_start)

    # store train loss
    t0 = datetime.now()
    train_losses.append(get_loss(usermovie2rating))

    # Store test loss
    test_losses.append(get_loss(usermovie2rating_test))
    print("Calculate cost:", datetime.now() - t0)
    print("Train loss:", train_losses[-1])
    print("Test loss:", test_losses[-1])


print("Train losses:", train_losses)
print("Test losses:", test_losses)


# Plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()