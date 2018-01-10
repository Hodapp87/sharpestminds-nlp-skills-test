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
            data = data.lower()
            l.append(data)
    return l

def load_imdb_data():
    """Load the IMDB review data.  The data can be downloaded here:
    http://ai.stanford.edu/~amaas/data/sentiment/

    Returned labels are 1 for a positive review, 0 for a negative
    review.  This also returns unsupervised training strings contained
    in the dataset; these are simply reviews not containing labels.

    Returns:
    X_train -- Training set reviews as a list of strings
    X_test -- Test set reviews as a list of strings
    y_train -- List of labels corresponding to X_train
    y_test -- List of labels corresponding to X_test
    X_unsup -- Unsupervised reviews as a list of strings
    """
    train_pos = read_all("aclImdb/train/pos")
    train_neg = read_all("aclImdb/train/neg")
    test_pos = read_all("aclImdb/test/pos")
    test_neg = read_all("aclImdb/test/neg")

    # Combine positive/negative reviews, and create labels:
    X_train = train_pos + train_neg
    X_test = test_pos + test_neg
    y_train = [1]*len(train_pos) + [0]*len(train_neg)
    y_test =  [1]*len(test_pos) + [0]*len(test_neg)

    # Unsupervised reviews are by themselves:
    X_unsup = read_all("aclImdb/train/unsup")

    return X_train, X_test, y_train, y_test, X_unsup

def tokenize(text):
    """Tokenize and filter a text sample.

    Parameters:
    text -- string to be tokenized and filtered.

    Returns:
    tokens -- a list of the tokens/words in text.
    """
    tokens = nltk.word_tokenize(text)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokens = [w for w in tokens if w not in stopwords]
    return tokens

def make_embedding_matrix(texts, size, save_file=None):
    """Create a Word2Vec model from a list of text samples.  If a filename
    is given in 'save_file', then the model will also be written to
    this (from which 'load_embedding_matrix' may then load later).

    Parameters:
    texts -- a list of text samples containing the vocabulary words.
    size -- the size of the word-vectors.
    save_file -- optional filename to write Word2Vec model to

    Returns: Trained Word2Vec model from gensim
    """
    model = Word2Vec([tokenize(s) for s in texts], size, sorted_vocab=1)
    if save_file:
        model.save(save_file)
    return model

def load_embedding_matrix(filepath):
    """Load a pre-trained Word2Vec model from a file.

    Parameters:
    filepath -- Path to filename containing saved model.

    Returns: gensim Word2Vec model
    """
    return Word2Vec.load(filepath)

def encode_words(text, word2idx, seq_length):
    """Tokenize the given text, convert it to word indices by the given
    dictionary, pad it to some length, and return it as a NumPy array.

    Words not in word2idx are just ignored, and the sequence is
    truncated at 'seq_length'.

    Parameters:
    text -- Text string to encode
    word2idx -- Dictionary mapping words to word indices
    seq_length -- Maximum sequence length

    Returns:
    NumPy integer array of shape (seq_length,) containing word indices.
    """
    idxs = np.zeros((seq_length,), dtype=np.int32)
    pos = 0
    # Note list truncation below:
    for pos,word in enumerate(tokenize(text)[:seq_length]):
        idxs[pos] = word2idx.get(word, 1)
    return idxs

def generate_batches(data, labels, batch_size, max_seq_length,
                     word2idx):

    """Generate batches of data and labels.  This will indefinitely
    iterate through the data, re-shuffling it at each epoch, and
    yielding (batch_data, batch_labels) at each iteration, suitable
    for fit_generator and evaluate_generator in Keras.

    Parameters:
    data -- List of strings
    labels -- List of binary labels corresponding with 'data'
    batch_size -- Size of batch to generate
    max_seq_length -- Maximum number of words to handle in string
    word2idx -- Dictionary mapping words to word indices

    Yields:
    batch_data -- NumPy array with one batch of data; shape is
                  (batch_size, max_seq_length).
    batch_labels -- NumPy array with corresponding binary labels;
                    shape is (batch_size,).

    """
    num_samples = len(data)
    labels = np.array(labels)
    while True:
        i = 0
        # Since we need to shuffle data & labels identically, get a
        # list of shuffled indices:
        shuffle_idxs = np.random.choice(num_samples, num_samples, replace=False)
        while i < num_samples:
            # and for each batch, refer to these indices:
            batch_idxs = shuffle_idxs[i:(i + batch_size)]
            batch_data = np.zeros((len(batch_idxs), max_seq_length))
            for batch,j in enumerate(batch_idxs):
                batch_data[batch, :] = encode_words(
                    data[j], word2idx, max_seq_length)
            batch_labels = labels[batch_idxs]
            i += batch_size
            yield batch_data, batch_labels
