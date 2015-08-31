# based on original script by: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import pandas as pd
import numpy as np

# get train and test data
n = 100000 # read from train
train = pd.read_csv("../input/train.csv", nrows=n)

# feature selection: use only numeric features
#numeric = ['int16', 'int32', 'int64', 'float16'] #, 'float32', 'float64']
#train = train.select_dtypes(include=numeric)

# drop missing
train.dropna

# remove IDs from train and test
train.drop('ID', axis=1, inplace=True)
print("train: %s" % (str(train.shape)))

black_list = ['VAR_0532']

import re
import sys
from time import time
from collections import defaultdict

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import DictVectorizer, FeatureHasher


def n_nonzero_columns(X):
    """Returns the number of non-zero columns in a CSR matrix X."""
    return len(np.unique(X.nonzero()[1]))


def tokens(doc):
    """Extract tokens from doc.

    This uses a simple regex to break strings into tokens. For a more
    principled approach, see CountVectorizer or TfidfVectorizer.
    """
    return (tok.lower() for tok in re.findall(r"\w+", doc))


def token_freqs(doc):
    """Extract a dict mapping tokens from doc to their frequencies."""
    freq = defaultdict(int)
    for tok in tokens(doc):
        freq[tok] += 1
    return freq

n_features = 100

print("DictVectorizer")
t0 = time()
vectorizer = DictVectorizer()
vectorizer.fit_transform(token_freqs(d) for d in train)
duration = time() - t0
print("Found %d unique terms" % len(vectorizer.get_feature_names()))
print(train)
print()

print("FeatureHasher on frequency dicts")
t0 = time()
hasher = FeatureHasher(n_features=n_features)
X = hasher.transform(token_freqs(d) for d in train)
duration = time() - t0
print("Found %d unique terms" % n_nonzero_columns(X))
print()

print("FeatureHasher on raw tokens")
t0 = time()
hasher = FeatureHasher(n_features=n_features, input_type="string")
X = hasher.transform(tokens(d) for d in train)
duration = time() - t0
print("Found %d unique terms" % n_nonzero_columns(X))

'''
for row in train:
    #if row not in black_list:
    print(train[row].describe())
    print("\n")
'''