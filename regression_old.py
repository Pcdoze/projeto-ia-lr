import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import sklearn.preprocessing
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

def normalize(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val) * 2 - 1

df = pd.read_csv('tabela-fipe-historico-precos-teste.csv', header =0,delimiter=",")
#df.tail(20000).to_csv('tabela-fipe-historico-precos-teste.csv', index=False)

df = df.head(10000)

df_new = df.drop(df.columns[[0, 1, 2, 3]], axis =1).apply(pd.to_numeric, errors='coerce')

Y = df_new.iloc[:,3]

df_new = df_new.apply(normalize)

X = df_new.drop(df_new.columns[[3]],axis=1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0)

classifier = LogisticRegression(solver = "lbfgs", random_state= 0)
classifier.fit(X_train, Y_train)

joblib.dump(classifier, os.getcwd() + '/regression_model.pkl')
    
predicted_y = classifier.predict(X_test)

print(predicted_y)