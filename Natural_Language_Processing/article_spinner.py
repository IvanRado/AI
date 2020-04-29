# A very simplistic implementation of an article spinner using trigrams

import nltk 
import random
import numpy as np

from bs4 import BeautifulSoup

# Load the reviews
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features='html5lib')
positive_reviews = positive_reviews.findAll('review_text')

# Extract trigrams and insert into dictionary
# (w1, w3) is the key, [w2] are the values
trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

# Turn each array of middle-words into a probability vector
trigram_probabilities = {}
for k, words in trigrams.items():
    # Create a dictionary mapping words to counts
    if len(set(words)) > 1:
        # Only do this when there are different posibilities for a middler word
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in d.items():
            d[w] = float(c) / n
        trigram_probabilities[k] = d

def random_sample(d):
    # Choose a random sample from dictionary where values are the probabilities
    r = random.random()
    cumulative = 0
    for w, p in d.items():
        cumulative += p
        if r < cumulative:
            return w

def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print("Original:", s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2:
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print("Spun:")
    print(" ".join(tokens).replace(" .", ".").replace(" ;", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))

if __name__ == "__main__":
    test_spinner()