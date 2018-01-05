# Copyright (c) 2017 Yazabi Predictive Inc.

#################################### MIT License ####################################
#                                                                                   #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
#                                                                                   #
#####################################################################################

# This module contains the function signatures for data preprocessing on the IMDB
# movie review dataset. The code is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @sharpestminds, or email us at contact@sharpestminds.com.
from __future__ import print_function
from __future__ import generators
from __future__ import division

import pickle

# useful packages
import nltk
import nltk.corpus
import gensim
from gensim.models.word2vec import Word2Vec
import sklearn
import numpy as np

import random
import os

def read_all(basedir, l=None):
    """Reads every file from the given directory. Returns a list in which
    each element contains the contents of one file.  This does not
    enter directories recursively.
    """
    l = []
    for basename in os.listdir(basedir):
        fname = os.path.join(basedir, basename)
        with open(fname) as f:
            data = "".join(f.readlines())
            l.append(data)
    return l

def load_imdb_data(one_hot_labels=True):
    """Load the imdb review data
    The data can be downloaded here: http://ai.stanford.edu/~amaas/data/sentiment/

    params:
    :one_hot_labels: if True, encode labels in one-hot format.

    returns: X_train, X_test, y_train, y_test
    """
    train_pos = read_all("aclImdb/train/pos")
    train_neg = read_all("aclImdb/train/neg")
    test_pos = read_all("aclImdb/test/pos")
    test_neg = read_all("aclImdb/test/neg")
    
    X_train = train_pos + train_neg
    X_test = test_pos + test_neg
    # Assign labels:
    if one_hot_labels:
        pos = [(1,0)]
        neg = [(0,1)]
    else:
        pos = ["pos"]
        neg = ["neg"]
    y_train = pos*len(train_pos) + neg*len(train_neg)
    y_test =  pos*len(test_pos) + neg*len(test_neg)
    return X_train, X_test, y_train, y_test

def tokenize(text):
    """Tokenize and filter a text sample.
    Hint: nltk

    params:
    :text: string to be tokenized and filtered.

    returns:
    :tokens: a list of the tokens/words in text.
    """
    tokens = nltk.word_tokenize(text)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokens = [w for w in tokens if w not in stopwords]
    return tokens

def make_embedding_matrix(texts, size, save_matrix=None):
    """Create an embedding matrix from a list of text samples.  If a
    filename is given in 'save_matrix', then the embedding matrix will
    also be written to this (from which 'load_embedding_matrix' may
    then load later).

    params:
    :texts: a list of text samples containing the vocabulary words.
    :size: the size of the word-vectors.
    :save_matrix: optional filename to write embedding matrix to

    returns:
    :embedding_matrix: a dictionary mapping words to word-vectors (embeddings).

    """
    model = Word2Vec([tokenize(s) for s in texts], size)
    if save_matrix:
        pickle.dump(model.wv, open(save_matrix, "wb"))
    return model.wv

def load_embedding_matrix(filepath):
    """Load a pre-trained embedding matrix
    Hint: save and load your embeddings to save time tweaking your model.

    returns:
    :embedding_matrix: a dictionary mapping words to word-vectors (embeddings).
    """
    return pickle.load(open(filepath, "rb"))

def to_word_vectors(tokenized_samples, embedding_matrix, max_seq_length):
    """Convert the words in each sample into word-vectors.

    params:
    :tokenized_samples: a list of tokenized text samples.
    :embedding_matrix: a dictionary mapping words to word-vectors.
    :max_seq_length: the maximum word-length of the samples.
=
    returns: a matrix containing the word-vectors of the samples with size:
    (num_samples, max_seq_length, word_vector_size).
    """
    # Does anyone know a better way of this?
    # embedding_matrix.vector_size just returns None.
    word_vector_size = embedding_matrix[embedding_matrix.index2word[0]].size
    num_samples = len(tokenized_samples)
    mtx = np.zeros((num_samples, max_seq_length, word_vector_size))
    for i,tokens in enumerate(tokenized_samples):
        # Truncate at a certain length (otherwise zero-pad - by
        # default since we started with np.zeros):
        for j,token in enumerate(tokens[:max_seq_length]):
            if token in embedding_matrix:
                mtx[i,j,:] = embedding_matrix[token]
            # Just ignore words that aren't found?
            # TODO: Is that kosher?
    return mtx

def generate_batches(data, labels, batch_size, max_seq_length,
                     embedding_matrix, rng=None):
    """"Generate batches of data and labels.

    Parameters:
    data -- List of strings
    labels -- List of one-hot encoded labels, corresponding with 'data'
    batch_size -- Size of batch to generate
    max_seq_length -- Maximum number of word vectors to handle in string
    embedding_matrix -- A dictionary mapping words to word-vectors
    rng -- Optional RandomState for shuffling

    Returns: (batch of data, batch of labels).
    """
    num_samples = len(data)
    while True:
        i = 0
        # Since we need to shuffle data & labels identically, get a
        # list of shuffled indices:
        shuffle_idxs = np.random.choice(num_samples, num_samples, replace=False)
        while i < num_samples:
            # and for each batch, refer to these indices:
            batch_idxs = shuffle_idxs[i:(i + batch_size)]
            batch_data = to_word_vectors(
                [tokenize(data[i]) for i in batch_idxs],
                embedding_matrix,
                max_seq_length
            )
            batch_labels = np.array([labels[i][0] for i in batch_idxs])
            i += batch_size
            yield batch_data, batch_labels
