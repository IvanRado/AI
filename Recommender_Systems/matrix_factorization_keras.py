# Matrix factorization using keras
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

# Load in the data
df = pd.read_csv('../LargeFiles/edited_rating.csv')

N = df.userId.max() + 1  # Number of users
M = df.movie_idx.max() + 1  # Number of movies

# Split into train and test
df = shuffle(df)
cutoff = int(0.8*len(*df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# Initialize variables
K = 10  # Latent dimensionality
mu = df_train.rating.mean()
epochs = 15
reg = 0.  # Regularization penalty

# Keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(u)
m_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(m)

u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u)
m_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(m)
x = Dot(axes=2)([u_embedding, m_embedding])

X = Add()([x, u_bias, m_bias])
x = Flatten()(x)

model = Model(inputs=[u, m], outputs=x)
model.compile(
    loss='mse',
    optimizer=SGD(lr=0.8, momentum=0.9),
    metrics=['mse']
)

r = model.fit(
    x=[df_train.userId.values, df_train.movie_idx.values],
    y=df_train.rating.values-mu,
    epochs=epochs,
    batch_size=128,
    validation_data=(
        [df_test.userId.values, df_test.movie_idx.values],
        df_test.rating.values - mu
    )
)

# Plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# Plot mse
plt.plot(r.history['mean_squared_error'], label="train mse")
plt.plot(r.history['val_mean_squared_error'], label="test mse")
plt.legend()
plt.show()