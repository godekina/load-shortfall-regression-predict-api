"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------
    # Your preprocessing steps here
    df_train = feature_vector_df
    df_train = df_train.drop(['Unnamed: 0'], axis=1)

    categorical_cols = ['Valencia_wind_deg', 'Seville_pressure']
    df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols)

    # create a copy
    df_train_copy = df_train_encoded.copy(deep=True)

    # Replace the null values in Valencia_pressure with Madrid_pressure values on the same row.
    df_train_copy['Valencia_pressure'].fillna(df_train_copy['Madrid_pressure'], inplace=True)

    # Extracting year, month, and hour from the time column
    df_train_copy['time'] = pd.to_datetime(df_train_copy['time'])
    df_train_copy['year'] = df_train_copy['time'].dt.year
    df_train_copy['month'] = df_train_copy['time'].dt.month
    df_train_copy['day'] = df_train_copy['time'].dt.day
    df_train_copy['Hours'] = df_train_copy['time'].dt.hour

    # Drop unnecessary columns and fill missing values with 0
    df_train_copy.drop(['time'], axis='columns', inplace=True)
    df_train_copy.fillna(0, inplace=True)

    features = ['Madrid_wind_speed', 'Bilbao_rain_1h',
                'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
                'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
                'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
                'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
                'Seville_rain_1h', 'Bilbao_snow_3h', 'Barcelona_pressure',
                'Seville_rain_3h', 'Madrid_rain_1h', 'Barcelona_rain_3h',
                'Valencia_snow_3h',
                'Bilbao_pressure', 'Valencia_pressure', 'Madrid_pressure',
                'Valencia_temp', 'Seville_temp',
                'Valencia_humidity', 'Barcelona_temp', 'Bilbao_temp',
                'Madrid_temp', 'Hours', 'Valencia_wind_deg_level_10', 'Valencia_wind_deg_level_2',
                'Valencia_wind_deg_level_3', 'Valencia_wind_deg_level_4',
                'Valencia_wind_deg_level_5', 'Valencia_wind_deg_level_6',
                'Valencia_wind_deg_level_7', 'Valencia_wind_deg_level_8',
                'Valencia_wind_deg_level_9', 'Seville_pressure_sp10',
                'Seville_pressure_sp11', 'Seville_pressure_sp12',
                'Seville_pressure_sp13', 'Seville_pressure_sp14',
                'Seville_pressure_sp15', 'Seville_pressure_sp16',
                'Seville_pressure_sp17', 'Seville_pressure_sp18',
                'Seville_pressure_sp19', 'Seville_pressure_sp2',
                'Seville_pressure_sp20', 'Seville_pressure_sp21',
                'Seville_pressure_sp22', 'Seville_pressure_sp23',
                'Seville_pressure_sp24', 'Seville_pressure_sp25',
                'Seville_pressure_sp3', 'Seville_pressure_sp4', 'Seville_pressure_sp5',
                'Seville_pressure_sp6', 'Seville_pressure_sp7', 'Seville_pressure_sp8',
                'Seville_pressure_sp9', 'month', 'day']

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = feature_vector_df[features]
    # ------------------------------------------------------------------------

    return predict_vector


def load_model(path_to_model: str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""


def save_transformer(transformer, transformer_path):
    """Save the preprocessing transformer to the specified path.

    Parameters
    ----------
    transformer : Transformer object
        The preprocessing transformer object.
    transformer_path : str
        The path to save the preprocessing transformer.
    """
    with open(transformer_path, 'wb') as file:
        pickle.dump(transformer, file)


def load_transformer(transformer_path):
    """Load the preprocessing transformer from the specified path.

    Parameters
    ----------
    transformer_path : str
        The path to the saved preprocessing transformer.

    Returns
    -------
    Transformer object
        The loaded preprocessing transformer object.
    """
    with open(transformer_path, 'rb') as file:
        transformer = pickle.load(file)
    return transformer


def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
