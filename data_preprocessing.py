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
    pass


def generate_batches(data, labels, batch_size, embedding_matrix=None):
    """"Generate batches of data and labels.
    Hint: tokenizi

    returns: batch of data and labels. When an embedding_matrix is passed in,
    data is tokenized and returned as matrix of word vectors.
    """
    yield batch_data, batch_labels
