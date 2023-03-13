from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import normalize


# padronizando os dados entre  parâmetros estipulados

# MinMaxScaler
X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
normalizator = MinMaxScaler(feature_range=(0, 1))
# print(normalizator.fit_transform(X))

# StandardScaler - distribuição gauciana
X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
normalizator = StandardScaler()
# print(normalizator.fit_transform(X))

# MaxAbsScaler - divide cada elemento pelo valor máximo da coluna
X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
normalizator = MaxAbsScaler()
# print(normalizator.fit_transform(X))

# normalize (l1 - soma o módulo de todos os valores)
# normalize (l2 - raiz quadrada da soma dos quadrados)
# normalize (max - MaxAbsScaler)

X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
normalizator = normalize(X, norm="l1")  # l1, l2, max
print(normalizator)
