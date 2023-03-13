from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
import pandas as pd

pd.set_option("display.max_columns", 30)

file = load_breast_cancer()

x = pd.DataFrame(file.data, columns=[file.feature_names])
y = pd.Series(file.target)

# print(x.shape, y.shape)
# print(y.value_counts())

x_training, x_test, y_training, y_test = train_test_split(
    x, y, test_size=0.3, random_state=9
)

model = LogisticRegression(solver="liblinear", C=95, penalty="l1")
model.fit(x_training, y_training)

score = model.score(x_test, y_test)
# print("Acurácia", score)

prediction = model.predict(x_test)
# print(prediction)
matrix = confusion_matrix(y_test, prediction)
print(matrix)
# [verdadeiro positivo, falso positivo     ]
# [falso negativo     , verdadeiro negativo]
