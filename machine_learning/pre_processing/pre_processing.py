import pandas as pd

pd.set_option("display.max_columns", 42)
dados = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\energy-benchmarking.csv"
)


# modificando o tipo da coluna
dados["DataYear"] = dados["DataYear"].astype(object)
# print(dados.dtypes)  # pegando os tipos


# retirando NAN dos datasets
dados_2 = dados.dropna()
# print(dados_2.head(100))


# verificando se o dado é nulo (True/False)
is_null = dados.isnull()
# print(is_null.head(100))


# somando dados nulos
missing = is_null.sum()
# print(missing.head(100))


# percentual de nulos
null_percentage = is_null.sum() / len(dados["OSEBuildingID"]) * 100
# print(null_percentage.head(100))


# substituindo os dados faltantes
dados["Comments"].fillna("No comments", inplace=True)
# dados["Compliant"].fillna("xablau", inplace=True)


# estratégia utilizada é colocar a média ou mediana nos campos NAN
dados["ENERGYSTARScore"].fillna(
    dados["ENERGYSTARScore"].mean(),
    inplace=True  # mean = média
    #               median = mediana
)

print(dados.isnull().sum() / len(dados["OSEBuildingID"]))
