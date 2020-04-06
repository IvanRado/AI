# Example of convolution with an image to perform blurring
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the famous Lena image
img = mpimg.imread('lena.png')

# Display the image to see the before
plt.imshow(img)
plt.show()

# Convert image to black and white
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

# Create a Gaussian filter
W = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i, j] = np.exp(-dist / 50.)
W /= W.sum() # normalize the kernel

# Filter visualization
plt.imshow(W, cmap='gray')
plt.show()

# Perform the convolution
out = convolve2d(bw, W)
plt.imshow(out, cmap='gray')
plt.show()


# Making the output the same size as the input
out = convolve2d(bw, W, mode='same')
plt.imshow(out, cmap='gray')
plt.show()
print(out.shape)


# In color
out3 = np.zeros(img.shape)
print(out3.shape)
for i in range(3):
    out3[:,:,i] = convolve2d(img[:,:,i], W, mode='same')
# out3 /= out3.max() # can also do this if you didn't normalize the kernel
plt.imshow(out3)
plt.show() # does not look like anything

