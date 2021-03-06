# Variational autoencoder implementaion in Theano
import util
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from theano.tensor.shared_randomstreams import RandomStreams


class DenseLayer(object):
  def __init__(self, M1, M2, f=T.nnet.relu):
    self.W = theano.shared(np.random.randn(M1, M2) * np.sqrt(2.0 / M1))
    self.b = theano.shared(np.zeros(M2))
    self.f = f
    self.params = [self.W, self.b]

  def forward(self, X):
    return self.f(X.dot(self.W) + self.b)


class VariationalAutoencoder:
  def __init__(self, D, hidden_layer_sizes):
    # hidden_layer_sizes specifies the size of every layer in the encoder up to the final hidden layer Z
    # The decoder will have the reverse shape

    
    # Represents a batch of training data
    self.X = T.matrix('X')

    # Encoder
    self.encoder_layers = []
    M_in = D
    for M_out in hidden_layer_sizes[:-1]:
      h = DenseLayer(M_in, M_out)
      self.encoder_layers.append(h)
      M_in = M_out


    # For convenience, we'll refer to the final encoder size as M
    M = hidden_layer_sizes[-1]

    # The encoder's final layer output is unbounded so there is no activation function
    # Need 2 times as many units as specified by M_out since there needs to be M_out means + M_out variances
    h = DenseLayer(M_in, 2 * M, f=lambda x: x)
    self.encoder_layers.append(h)

    # Get the mean and variance / std dev of Z.
    # Note that the variance must be > 0
    # Add a small amount for smoothing.
    current_layer_value = self.X
    for layer in self.encoder_layers:
      current_layer_value = layer.forward(current_layer_value)
    self.means = current_layer_value[:, :M]
    self.stddev = T.nnet.softplus(current_layer_value[:, M:]) + 1e-6

    # Get a sample of Z
    self.rng = RandomStreams()
    eps = self.rng.normal((self.means.shape[0], M))
    self.Z = self.means + self.stddev * eps


    # Decoder
    self.decoder_layers = []
    M_in = M
    for M_out in reversed(hidden_layer_sizes[:-1]):
      h = DenseLayer(M_in, M_out)
      self.decoder_layers.append(h)
      M_in = M_out

    # The decoder's final layer should go through a sigmoid
    h = DenseLayer(M_in, D, f=T.nnet.sigmoid)
    self.decoder_layers.append(h)

    # Get the posterior predictive
    current_layer_value = self.Z
    for layer in self.decoder_layers:
      current_layer_value = layer.forward(current_layer_value)
    self.posterior_predictive_probs = current_layer_value

    # Take samples from X_hat
    # Call this the posterior predictive sample
    self.posterior_predictive = self.rng.binomial(
      size=self.posterior_predictive_probs.shape,
      n=1,
      p=self.posterior_predictive_probs
    )

    # Take sample from a Z ~ N(0, 1) and put it through the decoder
    # Call this the prior predictive sample
    Z_std = self.rng.normal((1, M))
    current_layer_value = Z_std
    for layer in self.decoder_layers:
      current_layer_value = layer.forward(current_layer_value)
    self.prior_predictive_probs = current_layer_value

    self.prior_predictive = self.rng.binomial(
      size=self.prior_predictive_probs.shape,
      n=1,
      p=self.prior_predictive_probs
    )

    # Prior predictive from input 
    Z_input = T.matrix('Z_input')
    current_layer_value = Z_input
    for layer in self.decoder_layers:
      current_layer_value = layer.forward(current_layer_value)
    prior_predictive_probs_from_Z_input = current_layer_value


    # Now build the cost
    kl = -T.log(self.stddev) + 0.5*(self.stddev**2 + self.means**2) - 0.5
    kl = T.sum(kl, axis=1)
    expected_log_likelihood = -T.nnet.binary_crossentropy(
      output=self.posterior_predictive_probs,
      target=self.X,
    )
    expected_log_likelihood = T.sum(expected_log_likelihood, axis=1)
    self.elbo = T.sum(expected_log_likelihood - kl)

    

    # Define the updates
    params = []
    for layer in self.encoder_layers:
      params += layer.params
    for layer in self.decoder_layers:
      params += layer.params

    grads = T.grad(-self.elbo, params)

    # RMSProp
    decay = 0.9
    learning_rate = 0.001

    # For RMSProp
    cache = [theano.shared(np.ones_like(p.get_value())) for p in params]

    new_cache = [decay*c + (1-decay)*g*g for p, c, g in zip(params, cache, grads)]

    updates = [
        (c, new_c) for c, new_c in zip(cache, new_cache)
    ] + [
        (p, p - learning_rate*g/T.sqrt(new_c + 1e-10)) for p, new_c, g in zip(params, new_cache, grads)
    ]


    # Now define callable functions
    self.train_op = theano.function(
      inputs=[self.X],
      outputs=self.elbo,
      updates=updates
    )

    # Returns a sample from p(x_new | X)
    self.posterior_predictive_sample = theano.function(
      inputs=[self.X],
      outputs=self.posterior_predictive,
    )

    # Returns a sample from p(x_new | z), z ~ N(0, 1)
    self.prior_predictive_sample_with_probs = theano.function(
      inputs=[],
      outputs=[self.prior_predictive, self.prior_predictive_probs]
    )

    # Return mean of q(z | x)
    self.transform = theano.function(
      inputs=[self.X],
      outputs=self.means
    )

    # Returns a sample from p(x_new | z), from a given z
    self.prior_predictive_with_input = theano.function(
      inputs=[Z_input],
      outputs=prior_predictive_probs_from_Z_input
    )


  def fit(self, X, epochs=30, batch_sz=64):
    costs = []
    n_batches = len(X) // batch_sz
    print("n_batches:", n_batches)
    for i in range(epochs):
      print("epoch:", i)
      np.random.shuffle(X)
      for j in range(n_batches):
        batch = X[j*batch_sz:(j+1)*batch_sz]
        c = self.train_op(batch)
        c /= batch_sz # just debugging
        costs.append(c)
        if j % 100 == 0:
          print("iter: %d, cost: %.3f" % (j, c))
    plt.plot(costs)
    plt.show()


def main():
  X, Y = util.get_mnist()
  # Convert X to binary variable
  X = (X > 0.5).astype(np.float32)

  vae = VariationalAutoencoder(784, [200, 100])
  vae.fit(X)

  # Plot reconstruction
  done = False
  while not done:
    i = np.random.choice(len(X))
    x = X[i]
    im = vae.posterior_predictive_sample([x]).reshape(28, 28)
    plt.subplot(1,2,1)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(im, cmap='gray')
    plt.title("Sampled")
    plt.show()

    ans = input("Generate another?")
    if ans and ans[0] in ('n' or 'N'):
      done = True

  # Plot output from random samples in latent space
  done = False
  while not done:
    im, probs = vae.prior_predictive_sample_with_probs()
    im = im.reshape(28, 28)
    probs = probs.reshape(28, 28)
    plt.subplot(1,2,1)
    plt.imshow(im, cmap='gray')
    plt.title("Prior predictive sample")
    plt.subplot(1,2,2)
    plt.imshow(probs, cmap='gray')
    plt.title("Prior predictive probs")
    plt.show()

    ans = input("Generate another?")
    if ans and ans[0] in ('n' or 'N'):
      done = True


if __name__ == '__main__':
  main()
