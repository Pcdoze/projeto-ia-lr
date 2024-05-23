import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('tabela-fipe-historico-precos.csv', header =0,delimiter=",")

df = df.head(40000)

#df_new = df.drop(df.columns[[0, 1, 2, 3]], axis =1).apply(pd.to_numeric, errors='coerce')

#Y = df_new.iloc[:,3]
#X = df_new.drop(df_new.columns[[3]],axis=1)



newdf_data = []
newdf_dict = {}

for index, row in df.iterrows():
    # Access row data using row['column_name'] or row[index]
    
    if row['codigoFipe'] in newdf_dict:
        newdf_data.append({
            'codigoFipe': row['codigoFipe'],
            'modelo': row['modelo'],
            'anoModelo1': newdf_dict[row['codigoFipe']]['anoModelo'],
            'anoReferencia1': newdf_dict[row['codigoFipe']]['anoReferencia'],
            'valor1': newdf_dict[row['codigoFipe']]['valor'],
            'anoModelo2': row['anoModelo'],
            'anoReferencia2': row['anoReferencia'],
            'valor2': row['valor'],
        })
    else:
        newdf_dict[row['codigoFipe']] = {
            'codigoFipe': row['codigoFipe'],
            'modelo': row['modelo'],
            'anoModelo': row['anoModelo'],
            'anoReferencia': row['anoReferencia'],
            'valor': row['valor']
        }

df_new = pd.DataFrame(newdf_data)
df_new.to_csv('tabela-fipe-historico-precos-adaptada.csv', index=False)