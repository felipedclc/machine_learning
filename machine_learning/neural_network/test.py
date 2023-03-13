import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


# Carregue seus dados históricos de vendas
data = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\vw_dataset_for_ml_predict.csv"
)

# print(df.isnull().sum() / len(df["DATE"]) * 100)


data.drop("HOLIDAY", axis=1, inplace=True)
data.drop("COUNT_ACTIVE_USERS", axis=1, inplace=True)

# Separe suas variáveis de entrada (X) e saída (y)
X = data.drop("TOTAL_AMOUNT", axis=1)
y = data["TOTAL_AMOUNT"]

# Normalize seus dados para melhorar a precisão da rede neural
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()


# Inicialize o modelo sequencial
model = Sequential()

# Adicione as camadas da rede neural
model.add(Dense(32, activation="relu", input_shape=(X.shape[1],)))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="linear"))

# Compile o modelo
model.compile(loss="mse", optimizer="adam")


model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

# Crie uma nova entrada para fazer previsões
nova_entrada = np.array([[0.5, 0.3, 0.2]])

# Normalize a entrada
nova_entrada = (nova_entrada - X.mean()) / X.std()

# Faça a previsão usando a rede neural treinada
previsao = model.predict(nova_entrada)

# Desnormalize a previsão
previsao = previsao * y.std() + y.mean()

print(previsao)
