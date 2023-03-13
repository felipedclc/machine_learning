import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# PREVENDO O PREÇO DE VENDA DAS CASAS PARA CADA CARACTERÍSTICA


pd.set_option("display.max_columns", 21)
arquivo = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\kc_house_data.csv"
)

# EXCLUINDO FEATURES IRRELEVANTES
arquivo.drop("id", axis=1, inplace=True)
arquivo.drop("date", axis=1, inplace=True)
arquivo.drop("zipcode", axis=1, inplace=True)
arquivo.drop("lat", axis=1, inplace=True)
arquivo.drop("long", axis=1, inplace=True)
arquivo["sqft_above"].fillna(arquivo["sqft_above"].mean(), inplace=True)

# VARIÁVEIS PREDITORAS E VARIÁVEL TARGET
y = arquivo["price"]  # pegando preço
x = arquivo.drop("price", axis=1)  # pegando tudo menos o preço
# print(x.head())


# SEPARANDO OS DADOS EM TREINO E TESTE E CRIANDO OS DATASETS
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.3, random_state=14
)  # pegando 30% das linhas para teste / 70% treino


# CRIANDO O MODELO
modelo = Lasso(
    alpha=100, max_iter=2000, tol=0.001
)  # quanto maior o aplha mais generalista, padrão=1
modelo.fit(x_treino, y_treino)

# CALCULANDO O COEFICIENTE R2
resultado = modelo.score(x_teste, y_teste)  # score irá calcular R2
print(resultado)


# a Ridge regression tenta regularizar as variáveis e tirar as discrepâncias entre os maiores e menores valores

# Ridge = L2
# Lasso = L1
