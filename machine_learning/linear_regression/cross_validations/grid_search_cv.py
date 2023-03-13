from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import pandas as pd


# DIFERENÇA DO GRID X RANDOMIZED - GRID FAZ TODAS AS ITERAÇÕES, RANDOMIZED FAZ O NÚMERO QUE COLOCARMOS (n_iter)

pd.set_option("display.max_columns", 21)
arquivo = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\Admission_Predict.csv"
)

# print(arquivo.isnull().sum())
arquivo.drop("Serial No.", axis=1, inplace=True)

y = arquivo["Chance of Admit "]
x = arquivo.drop("Chance of Admit ", axis=1)

values = {
    "alpha": [
        0.1,
        0.5,
        1,
        2,
        5,
        10,
        25,
        50,
        100,
        150,
        200,
        300,
        500,
        750,
        1000,
        1500,
        2000,
        3000,
        5000,
    ],
    "l1_ratio": [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

model = ElasticNet()
search = GridSearchCV(
    estimator=model,
    param_grid=values,
    cv=5,
)

search.fit(x, y)

print("Melhor score:", search.best_score_)
print("Melhor alpha:", search.best_estimator_.alpha)
print("Melhor l1_ratio:", search.best_estimator_.l1_ratio)
