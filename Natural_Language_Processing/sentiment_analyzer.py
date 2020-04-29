# Simple sentiment analyzer implementation
# Dataset is a set of cleaned Amazon reviews on electronics

import nltk
import numpy as np
from sklearn.utils import shuffle

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# Load the reviews
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features='html5lib')
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features='html5lib')
negative_reviews = negative_reviews.findAll('review_text')

def my_tokenizer(s):
    s = s.lower()  # downcase for consistency
    tokens = nltk.tokenize.word_tokenize(s)  # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2]  # remove short words, they're unlikely to be useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # get the base form of words
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    return tokens

# Create a word-to-index map so that we can create our word-frequency vectors later
# We'll also save a tokenized version so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

for review in positive_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

print("Length of the word_index_map", len(word_index_map))

# Create our input matrices
def tokens_to_vectors(tokens, label):
    x = np.zeros(len(word_index_map) + 1)  # Last element is left for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()  # Normalize the feature vector before setting the label
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_index_map) + 1))
i = 0

for tokens in positive_tokenized:
    xy = tokens_to_vectors(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vectors(tokens, 0)
    data[i,:] = xy
    i += 1

# Shuffle the data then create train/test splits
orig_reviews, data = shuffle(orig_reviews, data)

X = data[:, :-1]
Y = data[:, -1]

# Reserve the last 100 rows to be the test data
X_train = X[:-100,]
Y_train = Y[:-100,]
X_test = X[-100:, ]
Y_test = Y[-100:, ]

model = LogisticRegression()
model.fit(X_train, Y_train)
print("Train accuracy:", model.score(X_train, Y_train))
print("Test accuracy:", model.score(X_test, Y_test))

# Look at words with weights that exceed 0.5
threshold = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)

# Check the misclassified examples
preds = model.predict(X)
P = model.predict_proba(X)[:, 1]  # Yields p(y = 1 | x)

# We will just print those that are the "most" wrong
minP_whenYis1 = 1
maxP_whenYis0 = 0
wrong_positive_review = None
wrong_negative_review = None
wrong_positive_prediction = None
wrong_negative_prediction = None

for i in range(N):
    p = P[i]
    y = Y[i]
    if y == 1 and p < 0.5:
        if p < minP_whenYis1:
            wrong_positive_review = orig_reviews[i]
            wrong_positive_prediction = preds[i]
            minP_whenYis1 = p
        elif y == 0 and p > 0.5:
            wrong_negative_review = orig_reviews[i]
            wrong_negative_prediction = preds[i]
            maxP_whenYis0 = p

print("Most wrong positive review (prob = %s, pred = %s):" % (minP_whenYis1, wrong_positive_prediction))
print(wrong_positive_review)
print("Most wrong negative review (prob = %s, pred = %s):" % (maxP_whenYis0, wrong_negative_prediction))
print(wrong_negative_review)