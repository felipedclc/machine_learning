from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

pd.set_option("display.max_columns", 30)

file = load_breast_cancer()

x = pd.DataFrame(file.data, columns=[file.feature_names])
y = pd.Series(file.target)

# normalizando variáveis preditoras
norm = MinMaxScaler(feature_range=(0, 1))
x_norm = norm.fit_transform(x)

# separando dados entre treino e teste
x_training, x_test, y_training, y_test = train_test_split(
    x_norm, y, test_size=0.3, random_state=16
)

# criando modelo
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_training, y_training)

# score
result = model.score(x_test, y_test)
print("Acurácia:", result)
