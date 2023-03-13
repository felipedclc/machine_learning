from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd


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


# separando variáveis entre preditoras e variável target
y = file["Instant.Liking"]
x = file.drop("Instant.Liking", axis=1)


# separando os dados em folds
stratifiedKFold = StratifiedKFold(n_splits=5)

# criando modelo
model = LogisticRegression(
    penalty="l2",
    solver="liblinear",  # liblinear com regularização L2 para arquivos pequenos
)
result = cross_val_score(model, x, y, cv=stratifiedKFold)

# imprimindo a acurácia
print(result.mean())
