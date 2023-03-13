from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import pandas as pd


def regression_models(x, y):
    reg = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elastic = ElasticNet()
    kfold = KFold(n_splits=5)

    result_reg = cross_val_score(reg, x, y, cv=kfold)
    result_ridge = cross_val_score(ridge, x, y, cv=kfold)
    result_lasso = cross_val_score(lasso, x, y, cv=kfold)
    result_elastic = cross_val_score(elastic, x, y, cv=kfold)

    # print("Regressão Linear: ", result_reg)
    # print("Regressão Ridge: ", result_ridge)
    # print("Regressão Lasso: ", result_lasso)
    # print("Regressão Elastic: ", result_elastic)

    models = []

    models.append({"Regressão Linear": result_reg.mean()})
    models.append({"Regressão Ridge": result_ridge.mean()})
    models.append({"Regressão Lasso": result_lasso.mean()})
    models.append({"Regressão Elastic": result_elastic.mean()})

    max_value = 0
    best_model = ""

    for model in models:
        for key, value in model.items():
            if value > max_value:
                max_value = value
                best_model = key

    print(best_model, max_value)


pd.set_option("display.max_columns", 21)
arquivo = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\Admission_Predict.csv"
)

# print(arquivo.isnull().sum())
arquivo.drop("Serial No.", axis=1, inplace=True)

y = arquivo["Chance of Admit "]
x = arquivo.drop("Chance of Admit ", axis=1)

# print(x.head())

regression_models(x, y)
