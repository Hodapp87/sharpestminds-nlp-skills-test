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

# This module contains a class template for building machine learning models
# on the IMDB movie review dataset. The code is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @sharpestminds, or email us at contact@sharpestminds.com.
from __future__ import print_function
from __future__ import generators
from __future__ import division

# import tensorflow as tf   # (optional) feel free to build your models using keras

import data_preprocessing as dp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression

import sklearn.metrics
import sklearn.pipeline
import numpy as np

import keras
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten

## recommended for LSTMTextClassifier
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM

class LSATextClassifier(object):
    """Text classifier using Latent Semantic Analysis (LSA) based on
    sklearn's TfidfVectorizer and TrunatedSVD.
    """

    def __init__(self, n_components = 175):
        """Initialize this classifier. This will compute LSA on the dataset,
        using n_components as the number of dimensions to reduce it to.

        Parameters:
        n_components -- Number of components to use for SVD (default 175)
        """
        self.n_components = n_components

    def build(self):
        """Build the model/graph."""
        self.pipeline = sklearn.pipeline.make_pipeline(
                # Get TF-IDF matrix:
                TfidfVectorizer(),
                # Perform LSA by doing SVD on this:
                TruncatedSVD(n_components=self.n_components),
                # Use this (reduced-dimensionality, dense) matrix to
                # train a classifier:
                LogisticRegression(C=0.5),
        )

    def train(self, train_data, train_labels):
        """Train the model on the training data."""
        y = np.array(train_labels)[:,0]
        self.pipeline.fit(train_data, y)

    def evaluate(self, test_data, test_labels):
        """Evaluate the model on the test data.

        returns:
        :accuracy: the model's accuracy classifying the test data.
        """
        y_predict = self.pipeline.predict(test_data)
        y_target = np.array(test_labels)[:,0]
        accuracy = sklearn.metrics.accuracy_score(y_target, y_predict)
        return accuracy

    def predict(self, review):
        """Predict the sentiment of an unlabelled review.

        returns: the predicted label of :review:
        """
        predict = self.pipeline.predict([review])
        return predict[0]

class CNNTextClassifier(object):
    """Text classifier based on a convolutional neural net in Keras.
    """

    def __init__(self, embedding_matrix=None):
        """Initialize the classifier with an (optional) embedding_matrix
        and/or any other parameters.
        """
        self.embedding_matrix = embedding_matrix
        self.model = None

    def build(self, word_vector_size, max_seq_length, features=128, depth=5):
        """Build the model.

        Parameters:
        word_vector_size -- Length of word vectors
        max_seq_length -- Maximum size of a word vector sequence
        features -- Number of features in hidden layers (default 128)
        depth -- Depth of hidden layers (default 5)
        """
        # Model is based around:
        # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        m = keras.models.Sequential()
        m.add(Conv1D(features, depth, activation="relu",
                     input_shape=(max_seq_length, word_vector_size)))
        m.add(MaxPooling1D(depth))
        m.add(Conv1D(features, depth, activation="relu"))
        m.add(MaxPooling1D(depth))
        m.add(Conv1D(features, depth, activation="relu"))
        m.add(MaxPooling1D(35))
        m.add(Flatten())
        m.add(Dense(128, activation="relu"))
        m.add(Dense(2, activation="softmax"))

        self.model = m

    def train(self, train_data, train_labels, batch_size=50, num_epochs=5, additional_parameters=None):
        """Train the model on the training data."""
        pass


    def evaluate(self, test_data, test_labels, additional_parameters=None):
        """Evaluate the model on the test data.

        returns:
        :accuracy: the model's accuracy classifying the test data.
        """

        pass


    def predict(self, review):
        """Predict the sentiment of an unlabelled review.

        returns: the predicted label of :review:
        """
        pass

class RNNTextClassifier(object):
    """Fill out this template to create three classes:
    LSATextClassifier(object)
    CNNTextClassifier(object)
    LSTMTextClassifier(object)

    Modify the code as much as you need.
    Add arguments to the functions and add as many other functions/classes as you wish.
    """

    def __init__(self, embedding_matrix=None, additional_parameters=None):
        """Initialize the classifier with an (optional) embedding_matrix
        and/or any other parameters."""
        self.embedding_matrix = embedding_matrix


    def build(self, model_parameters=None):
        """Build the model/graph."""
        pass


    def train(self, train_data, train_labels, batch_size=50, num_epochs=5, additional_parameters=None):
        """Train the model on the training data."""
        pass


    def evaluate(self, test_data, test_labels, additional_parameters=None):
        """Evaluate the model on the test data.

        returns:
        :accuracy: the model's accuracy classifying the test data.
        """

        pass


    def predict(self, review):
        """Predict the sentiment of an unlabelled review.

        returns: the predicted label of :review:
        """
        pass
