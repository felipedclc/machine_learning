from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# criando objeto OneHotEncoder
encoder = OneHotEncoder()

file = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\all_emissions.csv"
)

file.dropna(inplace=True)

# definindo os valores que serão testados em LogisticRegression
c_values = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
regularization = ["l1", "l2"]
grid_values = {"C": c_values, "penalty": regularization}
groups = np.array([1, 1, 2, 2])
# separando variáveis entre preditoras e variável target
y = file["TOTAL_AMOUNT"]
X = file.drop("TOTAL_AMOUNT", axis=1)

# ajustando e transformando os dados
X_encoded = encoder.fit_transform(X)

# print(logo)

# criando modelo
model = LogisticRegression(solver="liblinear")

# criando os grids
logistic_grid = GridSearchCV(
    estimator=model,
    param_grid=grid_values,
    scoring="accuracy",
    cv=10,
    verbose=1,
)
logistic_grid.fit(X_encoded, y)

# imprimindo a melhor acurácia e os melhores parâmetros
print("Melhor acurácia: ", logistic_grid.best_score_)
print("Parâmetro C: ", logistic_grid.best_estimator_.C)
print("Regularização: ", logistic_grid.best_estimator_.penalty)
