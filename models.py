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

## recommended for LSATextClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import Normalizer
# from sklearn.linear_model import LogisticRegression

## recommended for LSTMTextClassifier
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM

class ClassifierTemplate(object):
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
