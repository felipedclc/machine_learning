from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()

x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

# normalizando variáveis preditoras
norm = MinMaxScaler(feature_range=(0, 1))
x_norm = norm.fit_transform(x)

# definindo os valores que serão testados no KNN
k_values = np.array([3, 5, 7, 9, 11])
calc_distance = ["minkowski", "chebyshev"]
p_values = np.array([1, 2, 3, 4])
grid_values = {"n_neighbors": k_values, "metric": calc_distance, "p": p_values}

# criando modelo
model = KNeighborsClassifier()

# criando os grids
grid_KNN = GridSearchCV(estimator=model, param_grid=grid_values, cv=5)
grid_KNN.fit(x_norm, y)

# imprimindo os melhores parâmetros
print("Melhor acurácia:", grid_KNN.best_score_)
print("Melhor K:", grid_KNN.best_estimator_.n_neighbors)
print("Melhor distância:", grid_KNN.best_estimator_.metric)
print("Melhor valor p:", grid_KNN.best_estimator_.p)
