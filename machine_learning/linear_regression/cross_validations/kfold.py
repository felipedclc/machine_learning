import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# VALIDAÇÃO CRUZADA - SÃO CRIADOS CONJUNTOS QUE SERÃO TESTADOS DE FORMAS DIVERSAS

pd.set_option("display.max_columns", 21)
arquivo = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\Admission_Predict.csv"
)

# print(arquivo.isnull().sum())
arquivo.drop("Serial No.", axis=1, inplace=True)

y = arquivo["Chance of Admit "]
x = arquivo.drop("Chance of Admit ", axis=1)

model = LinearRegression()
kfold = KFold(n_splits=5)
result = cross_val_score(model, x, y, cv=kfold)

print(result.mean())
