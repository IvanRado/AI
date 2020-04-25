# Implementation of a simple RNN
# Tensorflow has basically defaulted to Keras for this so 
# implementation is done only using Keras API
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.layers import SimpleRNNCell
from keras.models import Sequential
from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels, all_parity_pairs

# M is the hidden layer size
def BasicRNN(X, M):

    N, T, D = X.shape

    model = Sequential()
    model.add(SimpleRNN(units=M, activation='relu', input_shape=(12, 1), return_sequences = False))
    model.add(Dense(units=12, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def parity(B =12, learning_rate=1., epochs=1000):
    X, Y = all_parity_pairs_with_sequence_labels(B)
    N, T, D = X.shape
    
    rnn = BasicRNN(X, 4)
    rnn.fit(X, Y, batch_size=100, epochs=1000)

if __name__ == '__main__':
    parity()
