import json
import joblib
import pandas as pd
import numpy as np
from azureml.core.model import Model
from sklearn.preprocessing import MaxAbsScaler

# Called when the service is loaded
def init():
    global bionelessModel
    global transformer
    global translater
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('bionelessModel_SWR')
    bionelessModel = joblib.load(model_path)
    transformer_path = Model.get_model_path('transformer_SWR')
    transformer = joblib.load(transformer_path)
    translater_path = Model.get_model_path('translater_SWR')
    translater = joblib.load(translater_path)
    
def prepareData(data):
    data = transformer.transform(data)
    return data

# Called when a request is received
def run(raw_data):
    # Get the input data as pandas Dataframe
    newData = pd.read_json(raw_data, orient='values')
    
    # Prepare & Normalize the data
    data = prepareData(newData)
    
    # Get a prediction from the model
    predictions = bionelessModel.predict(data, num_iteration=2612)
    translated_predictions = translater.inverse_transform(predictions)
    
    # Serialize predictions
    unique, counts = np.unique(translated_predictions, return_counts=True)
    predictionsNumber = dict(zip(unique, counts))
    
    for cle, valeur in predictionsNumber.items():
        predictionsNumber[cle] = str(valeur)
    
    predictionsJson = json.dumps(predictionsNumber)
    return predictionsJson