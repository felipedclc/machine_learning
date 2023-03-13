from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import pandas as pd


def regression_models(x_training, x_testing, y_training, y_testing):
    reg = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elastic = ElasticNet()

    reg.fit(x_training, y_training)
    ridge.fit(x_training, y_training)
    lasso.fit(x_training, y_training)
    elastic.fit(x_training, y_training)

    result_reg = reg.score(x_testing, y_testing)
    result_ridge = ridge.score(x_testing, y_testing)
    result_lasso = lasso.score(x_testing, y_testing)
    result_elastic = elastic.score(x_testing, y_testing)

    # print("Regressão Linear: ", result_reg)
    # print("Regressão Ridge: ", result_ridge)
    # print("Regressão Lasso: ", result_lasso)
    # print("Regressão Elastic: ", result_elastic)

    models = []

    models.append({"Regressão Linear": result_reg})
    models.append({"Regressão Ridge": result_ridge})
    models.append({"Regressão Lasso": result_lasso})
    models.append({"Regressão Elastic": result_elastic})

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

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.3, random_state=100
)

regression_models(x_treino, x_teste, y_treino, y_teste)
