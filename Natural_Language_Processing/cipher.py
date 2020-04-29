# A simple deciphering script using NLP

import numpy as np
import matplotlib.pyplot as plt


import string
import random
import re
import requests
import os
import textwrap

# Create a substitution cipher

# One will act as the key, other as the value
letters1 = list(string.ascii_lowercase)
letters2 = list(string.ascii_lowercase)

true_mapping = {}

# Shuffle the second set of letters
random.shuffle(letters2)

# Populate the map
for k, v in zip(letters1, letters2):
    true_mapping[k] = v

# Create our language model
# Initialize a Markov matrix
M = np.ones((26,26))

# Initial state distribution
pi = np.zeros(26)

# A function to update the Markov matrix
def update_transition(ch1, ch2):
    i = ord(ch1) - 97
    j = ord(ch2) - 97
    M[i,j] += 1

# A function to update the initial state distribution
def update_pi(ch):
    i = ord(ch) - 97
    pi[i] += 1

# Get the log probability of a word/token
def get_word_prob(word):
    i = ord(word[0]) - 97
    logp = np.log(pi[i])

    for ch in word[1:]:
        j = ord(ch) - 97
        logp += np.log(M[i, j])  # update the probability
        i = j  # Update j

    return logp

# Get the probability of a sequence of words
def get_sequence_prob(words):
    if type(words) == str:
        words = words.split()

    logp = 0
    for word in words:
        logp += get_word_prob(word)
    return logp

# For replacing non-alpha characters
regex = re.compile('[^a-zA-Z]')

# Load in words
for line in open('moby_dick.txt'):
    line = line.rstrip()

    # There are blank lines in the file
    if line:
        line = regex.sub(' ', line)  # Replace all non-alpha characters with space
        # Split the tokens in the line and lowercase
        tokens = line.lower().split()

        for token in tokens: 
            # Update the model
            ch0 = token[0]
            update_pi(ch0)

            for ch1 in token[1:]:
                update_transition(ch0, ch1)
                ch0 = ch1

    # Normalize the probabilities
pi /= pi.sum()
M /= M.sum(axis=1, keepdims=True)

original_message = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
'''

# A function to encode a message
def encode_message(msg):
    msg = msg.lower()
    msg = regex.sub(' ', msg)

    # Make the encoded message
    coded_msg = []
    for ch in msg:
        coded_ch = ch
        if ch in true_mapping:
            coded_ch = true_mapping[ch]
        coded_msg.append(coded_ch)

    return ''.join(coded_msg)

encoded_message = encode_message(original_message)

# A function to decode a message
def decode_message(msg, word_map):
    decoded_msg = []
    for ch in msg:
        decoded_ch = ch
        if ch in word_map:
            decoded_ch = word_map[ch]
        decoded_msg.append(decoded_ch)

    return ''.join(decoded_msg)

# Run an evolutionary algorithm to decode the message
# Start with the initialization
dna_pool = []
for _ in range(20):
    dna = list(string.ascii_lowercase)
    random.shuffle(dna)
    dna_pool.append(dna)

def evolve_offspring(dna_pool, n_children):
    # Makes n_children per offspring
    offspring = []

    for dna in dna_pool:
        for _ in range(n_children):
            copy = dna.copy()
            j = np.random.randint(len(copy))
            k = np.random.randint(len(copy))

            # Switch
            tmp = copy[j]
            copy[j] = copy[k]
            copy[k] = tmp
            offspring.append(copy)

    return offspring + dna_pool

num_iters = 1000
scores = np.zeros(num_iters)
best_dna = None
best_map = None
best_score = float('-inf')
for i in range(num_iters):
    if i > 0:
        # Get offspring from the current dna pool
        dna_pool = evolve_offspring(dna_pool, 3)

        # Calculate the score for each dna
        dna2score = {}
        for dna in dna_pool:
            # Populate the map
            current_map = {}
            for k, v in zip(letters1, dna):
                current_map[k] = v

            decoded_message = decode_message(encoded_message, current_map)
            score = get_sequence_prob(decoded_message)

            dna2score[''.join(dna)] = score

            # Record the best so far
            if score > best_score:
                best_dna = dna
                best_map = current_map
                best_score = score

        # Average score for this generation
        scores[i] = np.mean(list(dna2score.values()))

        # Keep the best 5 dna
        # Also turn them back into a list of single chars
        sorted_dna = sorted(dna2score.items(), key = lambda x: x[1], reverse=True)
        dna_pool = [list(k) for k, v in sorted_dna[:5]]

        # Print out some information every 200 iterations
        if i % 200 == 0:
            print("iter:", i, "score:", scores[i], "best so far:", best_score)


# Use the best score to decode
decoded_message = decode_message(encoded_message, best_map)
print("Log likelihood of decoded message:", get_sequence_prob(decoded_message))
print("Log likelihood of true message:", get_sequence_prob(regex.sub(' ', original_message.lower())))

# Which letters are wrong?
for true, v in true_mapping.items():
    pred = best_map[v]
    if true != pred:
        print("True:", true, "Pred:", pred)

# Print the final decoded message and the original message
print("Decoded message: \n", textwrap.fill(decoded_message))
print("\nTrue message: \n", original_message)

plt.plot(scores)
plt.show()