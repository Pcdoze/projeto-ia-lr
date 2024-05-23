import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def predictFipe(codigoFipe, 
                  modelo, 
                  anoModelo1,
                  anoReferencia1,
                  valor1,
                  anoModelo2,
                  anoReferencia2):
    
    df_data = []
    
    df_data.append({
            'codigoFipe': codigoFipe,
            'modelo': modelo,
            'anoModelo1': anoModelo1,
            'anoReferencia1': anoReferencia1,
            'valor1': valor1,
            'anoModelo2': anoModelo2,
            'anoReferencia2': anoReferencia2,
        })
    
    df = pd.DataFrame(df_data)
    
    df['diffAnos'] = df['anoReferencia2'] - df['anoReferencia1']
    df_new = df.drop(['codigoFipe', 'modelo'], axis =1).apply(pd.to_numeric, errors='coerce')

    model: LinearRegression = joblib.load(os.getcwd() + '/regression_model.pkl')
    predictions = model.predict(df_new)
    
    return predictions