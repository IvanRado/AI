# Simple convolution example demonstrating echo on a sound file
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

from scipy.io.wavfile import write

spf = wave.open("hellowworld.wav", "r")

signal = spf.readframes(-1)
signal = np.fromstring(signal, "Int16")
print("Numpu signal shape:", signal.shape)

plt.plot(signal)
plt.title("Hello world without echo")
plt.show()

delta = np.array([1., 0., 0.])
noecho = np.convolve(signal, delta)
print("noecho signal:", noecho.shape)
assert(np.abs(noecho[:len(signal)] - signal).sum() < 0.0000001)

noecho = noecho.astype(np.int16)
write('noecho.wav', 16000, noecho)

filt = np.zeros(16000)
filt[0] = 1
filt[4000] = 0.6
filt[8000] = 0.3
filt[1200] = 0.2
filt[1599] = 0.1
out = np.convolve(signal, filt)

out = out.astype(np.int16)
write('out.wav', 16000, out)