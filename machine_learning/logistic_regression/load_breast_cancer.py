from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np


pd.set_option("display.max_columns", 30)
file = load_breast_cancer()
x = pd.DataFrame(file.data, columns=[file.feature_names])  # variável preditora
y = pd.Series(file.target)  # variável target

# print(y.head(50))
# print(x.shape, y.shape)

# definindo os valores que serão testados em LogisticRegression
c_values = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
regularization = ["l1", "l2"]
grid_values = {"C": c_values, "penalty": regularization}

# criando modelo
model = LogisticRegression(solver="liblinear")

# criando os grids
logistic_grid = GridSearchCV(estimator=model, param_grid=grid_values, cv=5)
logistic_grid.fit(x, y)

# imprimindo a melhor acurácia e os melhores parâmetros
print("Melhor acurácia: ", logistic_grid.best_score_)
print("Parâmetro C: ", logistic_grid.best_estimator_.C)
print("Regularização: ", logistic_grid.best_estimator_.penalty)
