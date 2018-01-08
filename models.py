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

import numpy as np

# Random seed must be given before Keras imports if things are to be
# kept determinstic:
SEED = 0x12345
np.random.seed(SEED)

import data_preprocessing as dp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression

import sklearn.metrics
import sklearn.pipeline

import keras
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, \
    Dropout, Input, Embedding, LSTM
from keras.models import Sequential, Model

class LSATextClassifier(object):
    """Text classifier using Latent Semantic Analysis (LSA) based on
    sklearn's TfidfVectorizer and TruncatedSVD.
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

class IMDB_NN_Classifier(object):
    
    def __init__(self, embedding_matrix, word2idx, max_seq_length):
        self.embedding_matrix = embedding_matrix
        self.word2idx = word2idx
        self.max_seq_length = max_seq_length
        self.model = None

    def train(self, train_data, train_labels, batch_size=50, num_epochs=5):
        """Train the model on the training data."""
        # Split some data off for validation:
        num_samples = len(train_data)
        shuffle_idxs = np.random.choice(num_samples, num_samples, replace=False)
        split = int(num_samples * 0.8)
        train_idxs, valid_idxs = shuffle_idxs[:split], shuffle_idxs[split:]
        valid_data   = [train_data[i]   for i in valid_idxs]
        valid_labels = [train_labels[i] for i in valid_idxs]
        train_data   = [train_data[i]   for i in train_idxs]
        train_labels = [train_labels[i] for i in train_idxs]
        # Create separate training & validation generators:
        train_gen = dp.generate_batches(
            train_data, train_labels, batch_size, self.max_seq_length,
            self.word2idx)
        valid_gen = dp.generate_batches(
            valid_data, valid_labels, batch_size, self.max_seq_length,
            self.word2idx)
        self.model.fit_generator(
            train_gen,
            steps_per_epoch=len(train_data) / batch_size,
            epochs=num_epochs,
            validation_data=valid_gen,
            validation_steps=len(valid_data) / batch_size,
        )

    def evaluate(self, test_data, test_labels):
        """Evaluate the model on the test data.

        returns:
        :accuracy: the model's accuracy classifying the test data.
        """
        batch_size=32
        y = np.array(test_labels)[:,0]
        gen = dp.generate_batches(
            test_data, test_labels, batch_size, self.max_seq_length,
            self.word2idx)
        ev = self.model.evaluate_generator(
            gen,
            steps=len(test_data) / batch_size,
        )
        return ev[self.model.metrics_names.index("acc")]

    def predict(self, review):
        """Predict the sentiment of an unlabelled review.

        returns: the predicted label of :review:
        """
        idxs = dp.encode_words(review, self.word2idx, self.max_seq_length)
        y_predict = self.model.predict(idxs[np.newaxis, :])
        return y_predict[0,0]
    
class CNNTextClassifier(IMDB_NN_Classifier):
    """Text classifier based on a convolutional neural net in Keras.
    """

    def __init__(self, embedding_matrix, word2idx, max_seq_length):
        """Initialize the classifier with an (optional) embedding_matrix
        and/or any other parameters.

        Parameters:
        embedding_matrix -- NumPy array where row I is the word vector of the
                            word with index I. 
        word2idx -- Dictionary mapping word to word index
        max_seq_length -- Maximum size of a word vector sequence
        """
        super(CNNTextClassifier, self).__init__(
            embedding_matrix, word2idx, max_seq_length)

    def build(self, features=128, kernel=5):
        """Build the model.

        Parameters:
        features -- Number of features in hidden layers (default 128)
        kernel -- Convolution kernel size (default 5)
        """
        # Model is based around:
        # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        input_ = Input(shape=(self.max_seq_length,), dtype='int32')
        vocab_size, word_vector_size = self.embedding_matrix.shape
        x = Embedding(vocab_size, word_vector_size,
                      weights=[self.embedding_matrix],
                      input_length=self.max_seq_length,
                      trainable=False)(input_)
        x = Conv1D(features, kernel, activation="relu")(x)
        #m.add(Dropout(0.25))
        x = MaxPooling1D(kernel)(x)
        x = Conv1D(features, kernel, activation="relu")(x)
        #m.add(Dropout(0.25))
        x = MaxPooling1D(kernel)(x)
        x = Conv1D(features, kernel, activation="relu")(x)
        #m.add(Dropout(0.25))
        x = MaxPooling1D(35)(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(1, activation="sigmoid")(x)

        self.model = Model(inputs=[input_], outputs=[x])

        print(self.model.summary())
        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['acc'])

    def train(self, *args, **kw):
        super(CNNTextClassifier, self).train(*args, **kw)
        self.model.save_weights("cnn_text_classifier.h5")

class RNNTextClassifier(IMDB_NN_Classifier):
    """Classifier for IMDB reviews, based on an LSTM.
    """

    def __init__(self, embedding_matrix, word2idx, max_seq_length):
        """Initialize the classifier with an (optional) embedding_matrix
        and/or any other parameters.

        Parameters:
        embedding_matrix -- NumPy array where row I is the word vector of the
                            word with index I. 
        word2idx -- Dictionary mapping word to word index
        max_seq_length -- Maximum size of a word vector sequence
        """
        super(RNNTextClassifier, self).__init__(
            embedding_matrix, word2idx, max_seq_length)

    def build(self, units=128):
        """Build the model/graph.
        
        Parameters:
        units -- Number of output units for LSTM (default 128)
        """
        # Model is based around:
        # https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
        m = Sequential()
        vocab_size, word_vector_size = self.embedding_matrix.shape
        m.add(Embedding(vocab_size, word_vector_size,
                        weights=[self.embedding_matrix],
                        trainable=False))
        # To ignore word vectors:
        #m.add(Embedding(vocab_size, word_vector_size))
        m.add(LSTM(units, dropout=0.2, recurrent_dropout=0.2))
        m.add(Dense(1, activation="sigmoid"))

        self.model = m
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['acc'])
        print(self.model.summary())

    def train(self, *args, **kw):
        super(RNNTextClassifier, self).train(*args, **kw)
        self.model.save_weights("lstm_text_classifier.h5")
