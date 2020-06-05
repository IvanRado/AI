# Implementation of a Recursive Neural Network in Theano
import sys
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from sklearn.utils import shuffle
from util import init_weight, get_ptb_data, display_tree
from datetime import datetime

def adagrad(cost, params, lr, eps=1e-10):
    grads = T.grad(cost, params)
    caches = [theano.shared(np.ones_like(p.get_value())) for p in params]
    new_caches = [c + g*g for c, g in zip(caches, grads)]

    c_update = [(c, new_c) for c, new_c in zip(caches, new_caches)]
    g_update = [
        (p, p - lr*g / T.sqrt(new_c + eps)) for p, new_c, g in zip(params, new_caches, grads)
    ]

    updates = c_update + g_update
    return updates

class RecursiveNN:
    def __init__(self, V, D, K):
        self.V  # Vocab size
        self.D  # Dimensionality of the data
        self.K

    def fit(self, trees, learning_rate=3*1e-3, mu=0.99, reg=1e-4, epochs=15, activation=T.nnet.relu, train_inner_nodes=False):
        D = self.D
        V = self.V
        K = self.K
        self.f = activation
        N = len(trees)

        We = init_weight(V, D)
        Wh = np.random.randn(2, D, D) / np.sqrt(2 + D + D)
        bh = np.zeros(D)
        Wo = init_weight(D, K)
        bo = np.zeros(K)

        self.We = theano.shared(We)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wh, self.bh, self.Wo, self.bo]

        words = T.ivector('words')
        parents = T.ivector('parents')
        relations = T.ivector('relations')
        labels = T.ivector('labels')

    def recurrence(n, hiddens, words, parents, relations):
        # Any non-word will have index -1
        w = words[n]
        hiddens = T.switch(
            T.ge(w, 0),
            T.set_subtensor(hiddens[n], self.We[w]),
            T.set_subtensor(hiddens[n], self.f(hiddens[n] + self.bh))
        )

        r = relations[n]  # 0 is left, 1 is right
        p = parents[n]  # parent index
        hiddens = T.switch(
            T.ge(p, 0),
            T.set_subtensor(hiddens[p], hiddens[p] + hiddens[n].dot(self.Wh[r])),
            hiddens
        )
        return hiddens

    hiddens = T.zeros((words.shape[0], D))

    h, _ = theano.scan(
        fn=recurrence,
        outputs_info=[hiddens],
        n_steps=words.shape[0],
        sequences=T.arange(words.shape[0]),
        non_sequences=[words, parents, relations],
    )

    # The shape of h that is returned by scan is TxTxD
    # Because hiddens is TxD, and it does the recurrence T times
    # Technically this stores T times too much data
    py_x = T.nnet.softmax(h[-1].dot(self.Wo) + self.bo)

    prediction = T.argmax(py_x, axis = 1)

    rcost = reg*T.mean([(p*p).sum() for p in self.params])
    if train_inner_nodes:
        # Note this won't work for binary classification
        cost = -T.mean(T.log(py_x[T.arange(labels.shape[0], labels)])) + rcost
    else:
        cost = -T.mean(T.log(py_x[-1, labels[-1]])) + rcost

    updates - adagrad(cost, self.params, lr=8e-3)

    self.cost_predict_op = theano.function(
        inputs=[words, parents, relations, labels],
        outputs=[cost, prediction],
        allow_input_downcast=True,
    )

    self.train_op = theano.function(
        inputs=[words, parents, relationsh, labels],
        outputs=[h, cost, prediction],
        updates=updates
    )

    costs = []
    sequence_indexes = range(N)
    if train_inner_nodes:
        n_titak = sum(len(words) for words, _, _, in trees) 
    else:
        n_total = N
    for i in range(epochs):
        t0 = datetime.now()
        sequence_indexes = shuffle(sequence_indexes)
        n_correct = 0
        cost = 0
        it = 0
        for j in sequence_indexes:
            words, par, rel, lab = trees[j]
            _, c, p = self.train_op(words, par, rel, lab)

            if np.isnan(c):
                print("Cost is NaN; try decreasing the learning rate")
                exit()
            cost += c
            if train_inner_nodes:
                n_correct += np.sum(p == lab)
            else:
                n_correct += (p[-1] == lab[-1])
            it += 1
            if it % 1 == 0:
                sys.stdout.write("j/N: %d/%d correct rate so far: %f, cost so far: %f\r" % (it, N, float(n_correct)/n_total, cost))
                sys.stdout.flush()
        print(
            "i:", i, "cost:", cost,
            "correct rate:", (float(n_correct)/n_total),
            "time for epoch:", (datetime.now() - t0)
        )
        costs.append(cost)

    plt.plot(costs)
    plt.draw()  # Don't block later code

    def score(self, trees, idx2word=None):
        n_total = len(trees)
        n_correct = 0
        for words, par, rel, lab in trees:
            _, p = self.cost_predict_op(words, par, rel, lab)
            n_correct += (p[-1] == lab[-1])
        print("n_correct:", n_correct, "n_total:", n_total, end=" ")
        return float(n_correct) / n_total


def add_idx_to_tree(tree, current_idx):
    # Post-order labeling of tree nodes
    if tree is None:
        return current_idx
    current_idx = add_idx_to_tree(tree.left, current_idx)
    current_idx = add_idx_to_tree(tree.right, current_idx)
    tree.idx = current_idx
    current_idx += 1
    return current_idx


def tree2list(tree, parent_idx, is_binary=False, is_left=False, is_right=False):
    if tree is None:
        return [], [], [], []

    w = tree.word if tree.word is not None else -1
    if is_left:
        r = 0
    elif is_right:
        r = 1
    else:
        r = -1
    words_left, parents_left, relations_left, labels_left = tree2list(tree.left, tree.idx, is_binary, is_left=True)
    words_right, parents_right, relations_right, labels_right = tree2list(tree.right, tree.idx, is_binary, is_right=True)

    words = words_left + words_right + [w]
    parents = parents_left + parents_right + [parent_idx]
    relations = relations_left + relations_right + [r]
    if is_binary:
        if tree.label > 2:
            label = 1
        elif tree.label < 2:
            label = 0
        else:
            label = -1  # We will eventually filter these out
    else:
        label = tree.label
    labels = labels_left + labels_right + [label]

    return words, parents, relations, labels


def print_sentence(words, idx2word):
    for w in words:
        if w >= 0:
            print(idx2word[w], end=" ")


def main(is_binary=True):
    train, test, word2idx = get_ptb_data()

    for t in train:
        add_idx_to_tree(t, 0)
    train = [tree2list(t, -1, is_binary) for t in train]
    if is_binary:
        train = [t for t in train if t[3][-1] >= 0] # for filtering binary labels

    # sanity check
    # check that last node has no parent
    # for t in train:
    #     assert(t[1][-1] == -1 and t[2][-1] == -1)

    for t in test:
        add_idx_to_tree(t, 0)
    test = [tree2list(t, -1, is_binary) for t in test]
    if is_binary:
        test = [t for t in test if t[3][-1] >= 0] # for filtering binary labels

    train = shuffle(train)

    n_pos = sum(t[3][-1] for t in train)
    test = shuffle(test)
    test = test[:1000]

    V = len(word2idx)
    print("vocab size:", V)
    D = 10
    K = 2 if is_binary else 5

    model = RecursiveNN(V, D, K)
    model.fit(train, learning_rate=1e-2, reg=1e-2, mu=0, epochs=20, activation=T.tanh, train_inner_nodes=False)
    print("train accuracy:", model.score(train))
    print("test accuracy:", model.score(test))

    # make sure program doesn't end until we close the plot
    plt.show()


if __name__ == '__main__':
    main()
