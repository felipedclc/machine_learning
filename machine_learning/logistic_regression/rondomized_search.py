from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np


pd.set_option("display.max_rows", 64)
pd.set_option("display.max_columns", 64)

file = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\Data_train_reduced.csv"
)


file.drop("q8.2", axis=1, inplace=True)
file.drop("q8.8", axis=1, inplace=True)
file.drop("q8.9", axis=1, inplace=True)
file.drop("q8.10", axis=1, inplace=True)
file.drop("q8.17", axis=1, inplace=True)
file.drop("q8.18", axis=1, inplace=True)
file.drop("q8.20", axis=1, inplace=True)
file.drop("Product", axis=1, inplace=True)
file["q8.7"].fillna(file["q8.7"].median(), inplace=True)
file["q8.12"].fillna(file["q8.12"].median(), inplace=True)

file.drop("q1_1.personal.opinion.of.this.Deodorant", axis=1, inplace=True)
# essa variável foi deletada porque foi feito o feature selection que prevê interferência nos resultados

was_null = file.isnull().sum()
was_null_per = was_null / len(file["Product.ID"]) * 100
# print(was_null_per)
# print(file.shape)  # (linhas, colunas)
# print(file.dtypes)


# definindo os valores que serão testados em LogisticRegression
c_values = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
regularization = ["l1", "l2"]
grid_values = {"C": c_values, "penalty": regularization}

# separando variáveis entre preditoras e variável target
y = file["Instant.Liking"]
x = file.drop("Instant.Liking", axis=1)


# criando modelo
model = LogisticRegression(solver="liblinear")

# criando os grids
logistic_grid = GridSearchCV(estimator=model, param_grid=grid_values, cv=5)
logistic_grid.fit(x, y)

# imprimindo a melhor acurácia e os melhores parâmetros
print("Melhor acurácia: ", logistic_grid.best_score_)
print("Parâmetro C: ", logistic_grid.best_estimator_.C)
print("Regularização: ", logistic_grid.best_estimator_.penalty)
