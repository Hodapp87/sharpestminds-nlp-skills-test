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

def make_embedding_matrix(texts, size, save_file=None):
    """Create an embedding matrix & dictionary from a list of text
    samples.  If a filename is given in 'save_file', then the
    embedding matrix & dictionary will also be written to this (from
    which 'load_embedding_matrix' may then load later).

    In the returned values, the word index of 0 is reserved for no
    word.

    params:
    :texts: a list of text samples containing the vocabulary words.
    :size: the size of the word-vectors.
    :save_file: optional filename to write embedding matrix & dictionary to

    returns:
    :embedding_matrix: NumPy array where row I is the word vector of the
                       word with index I.
    :word2index: Dictionary mapping words to row indices in embedding_matrix
    """
    model = Word2Vec([tokenize(s) for s in texts], size)
    vocab_size = len(model.wv.vocab)
    word_vector_size = model[model.wv.index2word[0]].size
    array = np.zeros((vocab_size + 1, word_vector_size))
    word2idx = {}
    for i,word in enumerate(model.wv.vocab):
        array[i + 1, :] = model.wv[word]
        word2idx[word] = i + 1
    if save_file:
        pickle.dump((array, word2idx), open(save_file, "wb"))
    return array, word2idx

def load_embedding_matrix(filepath):
    """Load a pre-trained embedding matrix & dictionary.

    returns:
    :embedding_matrix: NumPy array where row I is the word vector of the
                       word with index I.
    :word2index: Dictionary mapping words to row indices in embedding_matrix
    """
    return pickle.load(open(filepath, "rb"))

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
    for word in tokenize(text)[:seq_length]:
        if word in word2idx:
            idxs[pos] = word2idx[word]
            pos += 1
    return idxs

def generate_batches(data, labels, batch_size, max_seq_length,
                     word2idx):

    """Generate batches of data and labels.  This will repeatedly iterate
    through the data, shuffling it at each epoch.  The batch of data
    will be of shape (batch_size, max_seq_length).

    Parameters:
    data -- List of strings
    labels -- List of one-hot encoded labels, corresponding with 'data'
    batch_size -- Size of batch to generate
    max_seq_length -- Maximum number of words to handle in string
    word2idx -- Dictionary mapping words to word indices

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
            batch_data = np.zeros((len(batch_idxs), max_seq_length))
            for batch,j in enumerate(batch_idxs):
                batch_data[batch, :] = encode_words(
                    data[j], word2idx, max_seq_length)
            batch_labels = np.array([labels[i][0] for i in batch_idxs])
            i += batch_size
            yield batch_data, batch_labels
