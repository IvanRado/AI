import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from util import getKaggleMNIST, error_rate
from keras.layers import Dense
from keras.models import Sequential
from sklearn.utils import shuffle

## Keras -------------------------

# # create random training data again
# Nclass = 500
# D = 10 # dimensionality of input
# M = 5 # hidden layer size
# K = 10 # number of classes

# X1 = np.random.randn(Nclass, D) + np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(X1[0])

# model = Sequential()
# model.add(Dense(units=M, activation='relu', input_shape=(D,)))
# model.add(Dense(units=K, activation='linear'))

# model.compile(loss='mean_squared_error', optimizer='rmsprop')
# hist = model.fit(x = X1, y= X1, batch_size=25, epochs=50)

# plt.plot(hist.history['loss'])
# plt.show()

# print("Actual values: {}".format(X1[0]))
# print("Predicted values: {}".format(model.predict(X1[0:1])))

## Tensorflow ------------------

class AutoEncoder(object):
    def __init__(self, D, M, an_id):
        self.M = M
        self.id = an_id
        self.build(D, M)
    
    def set_session(self, session):
        self.session = session

    def build(self, D, M):
        self.W = tf.Variable(tf.random_normal(shape=(D,M)))
        self.bh = tf.Variable(np.zeros(M).astype(np.float32))
        self.bo = tf.Variable(np.zeros(D).astype(np.float32))

        self.X_in = tf.placeholder(tf.float32, shape=(None, D))
        self.Z = self.forward_hidden(self.X_in)
        self.X_hat = self.forward_output(self.X_in)

        logits = self.forward_logits(self.X_in)
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = self.X_in,
                logits=logits,
            )
        )

        self.train_op = tf.train.AdamOptimizer(1e-1).minimize(self.cost)

    def fit(self, X, epochs=1, batch_size=100, show_fig=False):
        N, D = X.shape
        n_batches = N // batch_size
        costs = []
        print("Training autoencoder : %s" % self.id)
        for i in range(epochs):
            print("epoch:", i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_size:(j*batch_size + batch_size)]
                _, c = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: batch})
                if j % 10 == 0:
                    print("j / n_batches:", j, '/', n_batches, "cost:", c)
                costs.append(c)
            if show_fig:
                plt.plot(costs)
                plt.show()

    def transform(self, X):
        return self.session.run(self.Z, feed_dict={self.X_in: X})

    def predict(self, X):
        return self.session.run(self.X_hat, feed_dict={self.X_in: X})

    def forward_hidden(self, X):
        Z = tf.nn.sigmoid(tf.matmul(X, self.W) + self.bh)
        return Z

    def forward_logits(self, X):
        Z = self.forward_hidden(X)
        return tf.matmul(Z, tf.transpose(self.W)) + self.bo

    def forward_output(self, X):
        return tf.nn.sigmoid(self.forward_logits(X))

class DNN(object):
    def __init__(self, D, hidden_layer_sizes, K, UnsupervisedModel=AutoEncoder):
        self.hidden_layers = []
        count = 0
        input_size = D
        for output_size in  hidden_layer_sizes:
            ae = UnsupervisedModel(input_size, output_size, count)
            self.hidden_layers.append(ae)
            count += 1
            input_size = output_size
        self.build_final_layer(D, hidden_layer_sizes[-1], K)

    def set_session(self, session):
        self.session = session
        for layer in self.hidden_layers:
            layer.set_session(session)
    
    def build_final_layer(self, D, M, K):
        # Initialize logistic regression layer
        self.W = tf.Variable(tf.random_normal(shape=(M,K)))
        self.b = tf.Variable(np.zeros(K).astype(np.float32))

        self.X = tf.placeholder(tf.float32, shape = (None, D))
        labels = tf.placeholder(tf.int32, shape=(None,))
        self.Y = labels
        logits = self.forward(self.X)

        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost)
        self.prediction = tf.argmax(logits, 1)

    def fit(self, X, Y, Xtest, Ytest, pretrain=True, epochs = 1, batch_size=100):
        N = len(X)

        # Greedy layer-wise training of autoencoders
        pretrain_epochs = 1
        if not pretrain:
            pretrain_epochs = 0

        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, epochs=pretrain_epochs)
            # Create current_input for the next layer
            current_input = ae.transform(current_input)


        n_batches = N // batch_size
        costs = []

        print("Supervised training...")
        for i in range(epochs):
            print("Epoch:", i)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_size:(j*batch_size + batch_size)]
                Ybatch = Y[j*batch_size:(j*batch_size + batch_size)]
                self.session.run(
                    self.train_op,
                    feed_dict={self.X: Xbatch, self.Y: Ybatch}
                )

                c,p = self.session.run(
                    (self.cost, self.prediction),
                    feed_dict={self.X: Xtest, self.Y: Ytest}
                )
                error = error_rate(p, Ytest)
                if j % 10 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c, "error:", error)
                costs.append(c)
        plt.plot(costs)
        plt.show()

    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            Z = ae.forward_hidden(current_input)
            current_input = Z

        # Logistic layer
        logits = tf.matmul(current_input, self.W) + self.b
        return logits

def test_pretraining_dnn():
    X_train, Y_train, X_test, Y_test = getKaggleMNIST()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    _, D = X_train.shape
    K = len(set(Y_train))
    dnn = DNN(D, [1000, 750, 500], K)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        dnn.set_session(session)
        dnn.fit(X_train, Y_train, X_test, Y_test, pretrain=True, epochs = 10)

def test_single_autoencoder():
    X_train, Y_train, X_test, Y_test = getKaggleMNIST()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    _, D = X_train.shape
    autoencoder = AutoEncoder(D, 300, 0)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        autoencoder.set_session(session)
        autoencoder.fit(X_train, show_fig=True)

        # Compare the original images to those reconstructed by the autoencoder
        done = False
        while not done:
            i = np.random.choice(len(X_test))
            x = X_test[i]
            y = autoencoder.predict([x])
            
            plt.subplots(1,2,1)
            plt.imshow(x.reshape(28,28), cmap = 'gray')
            plt.title('Original')

            plt.subplot(1,2,2)
            plt.imshow(y.reshape(28,28), cmap='gray')
            plt.title('Reconstructed')

            plt.show()

            ans = input("Generate another?")
            if ans and ans[0] in ('n' or 'N'):
                done = True

if __name__ == '__main__':
    # Test a single autoencoder
    test_pretraining_dnn()
        

