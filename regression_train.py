import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib


df = pd.read_csv('tabela-fipe-historico-precos-adaptada.csv', header =0,delimiter=",")
df = df.head(10000)

df['diffAnos'] = df['anoReferencia2'] - df['anoReferencia1']

df_new = df.drop(['codigoFipe', 'modelo'], axis =1).apply(pd.to_numeric, errors='coerce')

Y = df_new['valor2']
X = df_new.drop('valor2',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Avaliando o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualização da regressão
# plt.scatter(X_test['diffAnos'], y_test, color='blue')
# plt.plot(X_test['diffAnos'], y_pred, color='red')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Linear Regression')
# plt.show()

joblib.dump(model, os.getcwd() + '/regression_model.pkl')