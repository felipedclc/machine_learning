from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import pandas as pd

iris = load_iris()

x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

x_training, x_test, y_training, y_test = train_test_split(
    x, y, test_size=0.3, random_state=67
)

model = GaussianNB()
model.fit(x_training, y_training)

score = model.score(x_test, y_test)
print("Acurácia:", score)
