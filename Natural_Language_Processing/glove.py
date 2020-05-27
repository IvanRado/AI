# Implementation of GLoVe word embedding algorithm
import numpy as np
import json 
import os
# import theano
# import theano.tensor as T
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.utils import shuffle
from word2Vec import get_wiki
from util import find_analogies

class Glove: 
    
    def __init__(self, D, V, context_sz):
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=10e-5, reg=0.1, xmax=100, alpha=0.75, epochs=10, gd=False, use_theano=True):
        t0 = datetime.now()
        V = self.V
        D = self.D

        if not os.path.exists(cc_matrix):
            X = np.zeros((V,V))
            N = len(sentences)
            print("Number of sentences to process:", N)
            it = 0
            for sentence in sentences:
                it += 1
                if it % 10000 == 0:
                    print("Processed", it, "/", N)

                n = len(sentence)
                for i in range(n):
                    # i, j are not the word indexes
                    wi = sentence[i]

                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_sz)

                    # Start and end tokens
                    if i - self.context_sz < 0:
                        points = 1.0 / (i+1)
                        X[wi,0] += points
                        X[0,wi] += points
                    if i + self.context_sz > n:
                        points = 1.0 / (n - 1)
                        X[wi,1] += points
                        X[1,wi] += points

                    # Left side
                    for j in range(start, i):
                        wj = sentence[j]
                        points = 1.0 / (i - j)
                        X[wi, wj] += points
                        X[wj, wi] += points

                    # Right side
                    for j in range(i + 1, end):
                        wj = sentence[j]
                        points = 1.0 / (j - i)
                        X[wi, wj] += points
                        X[wj, wi] += points

            np.save(cc_matrix, X)

        else:
            X = np.load(cc_matrix)

        print("Max value in X:", X.max())

        fX = np.zeros((V,V))
        fX[X < xmax] = (X[X < xmax] / float(xmax))**alpha
        fX[X >= xmax] = 1

        logX = np.log(X + 1)

        print("Time to build co-occurence matrix:", (datetime.now() - t0))

        # Initialize the weights
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()

        if gd and use_theano:
            pass

        costs = []
        sentence_indexes = range(len(sentences))
        for epoch in range(epochs):
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
            cost = (fX *delta *delta).sum()
            costs.append(cost)
            print("epoch:", epoch, "cost:", cost)

            if gd:
                if use_theano:
                    pass
                else:
                    oldW = W.copy()
                    for i in range(V):
                        W[i] -= learning_rate*(fX[i,:]*delta[i,:]).dot(U)
                    W -= learning_rate*reg*W

                    for i in range(V):
                        b[i] -= learning_rate*fX[i,:].dot(delta[i,:])
                    b -= learning_rate*reg*b

                    for j in range(V):
                        U[j] -= learning_rate*(fX[:,j]*delta[:,j]).dot(oldW)
                    U -= learning_rate*reg*U

                    for j in range(V):
                        c[j] -= learning_rate*fX[:,j].dot(delta[:,j])
                    c -= learning_rate*reg*c
            else:
                # Alternating least squares
                for i in range(V):
                    matrix = reg*np.eye(D) + (fX[i,:]*U.T).dot(U)
                    vector = (fX[i,:]*logX[i,:] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(matrix, vector)

                for i in range(V):
                    denominator = fX[i,:].sum()
                    numerator = fX[i,:].dot(logX[i,:] - W[i].dot(U.T) - c - mu)
                    b[i] = numerator/denominator / (1 + reg)

                for j in range(V):
                    matrix = reg*np.eye(D) + (fX[:,j]*W.T).dot(W)
                    vector = (fX[:,j]*logX[:,j] - b - c[j] - mu)).dot(U)
                    U[j] = np.lingalg.solve(matrix, vector)

                for j in range(V):
                    denominator = fX[:,j].sum()
                    numerator = fX[:,j].dot(logX[:,j] - W[j].dot(U.T) - b - mu)
                    c[j] = numerator/denominator / (1 + reg)

                

        self.W = W
        self.U = U

        plt.plot(costs)
        plt.show()

    def save(self, fn):
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)

            

def main(we_file, w2i_file, n_files=50):
    cc_matrix = "cc_matrix_%s.npy" % n_files

    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)
        sentences = []
    else:
        sentences, word2idx = get_wiki()
        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)

    V = len(word2idx)
    model = Glove(80, V, 10)
    model.fit(
        sentences = sentences,
        cc_matrix=cc_matrix,
        learning_rate=3*10e-5,
        reg = 0.01,
        epochs=2000,
        gd=True,
        use_theano=False,
    )

    model.save(we_file)


if __name__ == '__main__':
    we = 'glove_model_50.npz'
    w2i = 'glove_word2idx_50.json'
    main(we, w2i)
    for concat in (True, False):
        print("** concat:", concat)
        find_analogies('king', 'man', 'woman', concat, we, w2i)
        find_analogies('france', 'paris', 'london', concat, we, w2i)
        find_analogies('france', 'paris', 'rome', concat, we, w2i)
        find_analogies('paris', 'france', 'italy', concat, we, w2i)
        find_analogies('france', 'french', 'english', concat, we, w2i)
        find_analogies('japan', 'japanese', 'chinese', concat, we, w2i)
        find_analogies('japan', 'japanese', 'italian', concat, we, w2i)
        find_analogies('japan', 'japanese', 'australian', concat, we, w2i)
        find_analogies('december', 'november', 'june', concat, we, w2i)