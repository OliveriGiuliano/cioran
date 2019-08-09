import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
from keras.preprocessing.image import ImageDataGenerator
from .model import create_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
from keras.utils import np_utils

print("real", os.path.realpath(__file__))

TRAINING_DATA_FILE = os.path.dirname(os.path.realpath(__file__)) + '/dataset/train.txt'
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/saved_model/best_model.hdf5'


def _save_model(model):
    pickle.dump(model, open(MODEL_PATH, 'wb'))


def train_model():
    K.clear_session()

    # load ascii text and covert to lowercase
    filename = TRAINING_DATA_FILE
    raw_text = open(filename, encoding="UTF-8").read()
    raw_text = raw_text.lower()

    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)
    print("set cars: ", set(chars))

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)

    # define the checkpoint
    filepath = MODEL_PATH
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model = create_model(input_shape=(X.shape[1], X.shape[2]), output_shape=y.shape[1])
    model.fit(X, y, epochs=1, batch_size=64, callbacks=callbacks_list)
    K.clear_session()
    return 0


if __name__ == '__main__':
    print(train_model())
