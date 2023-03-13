from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
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
    best_model = []
    for model in models:
        for key, value in model.items():
            if value > max_value:
                max_value = value
                best_model.append(key)

    print(best_model[-1], max_value)

    if best_model[-1] == "Regressão Linear":
        return reg
    elif best_model[-1] == "Regressão Ridge":
        return ridge
    elif best_model[-1] == "Regressão Lasso":
        return lasso
    elif best_model[-1] == "Regressão Elastic":
        return elastic


# pd.set_option("display.max_columns", 2)
file = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\all_emissions.csv"
)

y = file["TOTAL_AMOUNT"]
x = file.drop("TOTAL_AMOUNT", axis=1)

norm = MinMaxScaler(feature_range=(0, 1))
x_norm = norm.fit_transform(x)

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x_norm, y, test_size=0.3
)

best_model = regression_models(x_treino, x_teste, y_treino, y_teste)

# file_2023 = pd.read_csv(
#     "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\all_emissions_2023.csv"
# )

# x2 = file_2023.drop("TOTAL_AMOUNT", axis=1)

# norm = MinMaxScaler(feature_range=(0, 1))
# x_norm2 = norm.fit_transform(x2)

y_pred = best_model.predict(x_teste)
error_metric = mean_squared_error(y_pred, y_teste)
# print("The mean square error this model is:", error_metric)

fig, ax = plt.subplots()
ax.scatter(y_teste, y_pred)
ax.plot(y_teste, y_teste, color="red")
ax.set_xlabel("Testing target values")
ax.set_ylabel("Predicted target values")
ax.set_title("Predicted vs atual values")

plt.show()
