# Application of a HMM to Parts-Of-Speeching Tagging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import sys
sys.path.append(os.path.abspath('..'))

from Hidden_Markov_Models.hmm_scaled import HMM
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import f1_score

def get_data(split_sequences=False):
    if not os.path.exists('chunking'):
        print("Please create a folder in your local directory called 'chunking'")
        print("train.txt and test.txt should be stored in there.")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists('chunking/train.txt'):
        print("train.txt is not in chunking/train.txt")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists('chunking/test.txt'):
        print("test.txt is not in chunking/test.txt")
        print("Please check the comments to get the download link.")
        exit()

    word2idx = {}
    tag2idx = {}
    word_idx = 0
    tag_idx = 0
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []
    for line in open('chunking/train.txt'):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])
            
            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []

    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY

    # load and score test data
    Xtest = []
    Ytest = []
    currentX = []
    currentY = []
    for line in open('chunking/test.txt'):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word in word2idx:
                currentX.append(word2idx[word])
            else:
                currentX.append(word_idx) # use this as unknown
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtest.append(currentX)
            Ytest.append(currentY)
            currentX = []
            currentY = []
    if not split_sequences:
        Xtest = currentX
        Ytest = currentY

    return Xtrain, Ytrain, Xtest, Ytest, word2idx

def accuracy(T, Y):
    # Inputs are lists of lists
    n_correct = 0 
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == Y)
        n_total += len(y)
    return float(n_correct) / n_total

def total_f1_score(T, Y):
    # Inputs are lists of lists
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()

def main(smoothing=1e-1):
    # X = words, Y = Parts-of-Speech tags
    X_train, y_train, X_test, y_test, word2idx = get_data(split_sequences=True)
    V = len(word2idx) + 1

    # Find hidden state transition matrix
    M = max(max(y) for y in y_train) + 1
    A = np.ones((M, M))*smoothing  # Add-one smoothing
    pi = np.zeros(M)
    for y in y_train:
        pi[y[0]] += 1
        for i in range(len(y) - 1):
            A[y[i], y[i+1]] += 1
    # Turn it into a probability matrix
    A /= A.sum(axis = 1, keepdims=True)
    pi /= pi.sum()

    # Find the observation matrix
    B = np.ones((M, V))*smoothing  # Add-one smoothing
    for x, y in zip(X_train, y_train):
        for xi, yi in zip(x, y):
            B[yi, xi] += 1
    B /= B.sum(axis = 1, keepdims=True)

    hmm = HMM(M)
    hmm.pi = pi
    hmm.A = A
    hmm.B = B

    # Get predictions
    P_train = []
    for x in X_train:
        p = hmm.get_state_sequence(x)
        P_train.append(p)

    P_test = [] 
    for x in X_test:
        p = hmm.get_state_sequence(x)
        P_test.append(p)

    # Print the results
    print("Train accuracy:", accuracy(y_train, P_train))
    print("Test accuracy:", accuracy(y_test, P_test))
    print("Train F1:", total_f1_score(y_train, P_train))
    print("Test F1:", total_f1_score(y_test, P_test))

if __name__ == "__main__":
    main()


