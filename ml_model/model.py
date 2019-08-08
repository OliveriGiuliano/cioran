from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
from keras import backend as K


def create_model():
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(100, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(41, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
