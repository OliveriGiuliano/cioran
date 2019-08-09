from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
import sys
from keras.preprocessing import image
import numpy as np
import pandas as pd
from ml_model.run_pipeline import train_model
from ml_model.prediction import make_prediction
from sqlalchemy import create_engine
from keras.models import model_from_json, load_model
from ml_model.model import create_model
from ml_model.run_pipeline import TRAINING_DATA_FILE
from keras import backend as K
# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
PORT = 5000 # os.environ['PORT']
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/ml_model/saved_model/best_model.h5'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)


# make a predict
@app.route('/predict', methods=['POST'])
def predict():
    K.clear_session()


    filename = TRAINING_DATA_FILE
    raw_text = open(filename, encoding="UTF-8").read()
    raw_text = raw_text.lower()
    chars = sorted(list(set(raw_text)))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    n_vocab = len(chars)
    n_chars = len(raw_text)
    seq_length = 100
    

    model = create_model(input_shape=(seq_length,1),output_shape=n_vocab)
    model.load_weights(MODEL_PATH)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    # pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")
    K.clear_session()
    return jsonify(pattern)


@app.route('/train', methods=['GET'])
def train():
    train_model()
    return "Training completed"


@app.route('/status', methods=['GET'])
def status():
    return "Everythign gucci"


# Run Server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
