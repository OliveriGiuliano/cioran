import pickle
import pandas as pd
from .run_pipeline import MODEL_PATH


def make_prediction(input_data, model_path=MODEL_PATH):
    if type(input_data) is dict:
        data = pd.DataFrame(input_data, index=[0])
    else:
        data = pd.DataFrame(input_data)

    loaded_model = pickle.load(open(model_path, 'rb'))
    result = loaded_model.predict(data)
    return result
