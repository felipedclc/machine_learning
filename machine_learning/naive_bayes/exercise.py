from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from time import time

pd.set_option("display.max_columns", 21)
dataset = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\wine_dataset.csv"
)

y = dataset["style"]
x = dataset.drop("style", axis=1)
start = time()
model = GaussianNB()
end = time()
print("tempo de treinamento:", end - start)

skfold = StratifiedKFold(n_splits=3)
result = cross_val_score(model, x, y, cv=skfold)
print("Acurácia:", result.mean())
