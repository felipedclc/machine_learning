from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", 30)
file = load_breast_cancer()

x = pd.DataFrame(file.data, columns=[file.feature_names])
y = pd.Series(file.target)

x_training, x_test, y_training, y_test = train_test_split(
    x, y, test_size=0.3, random_state=9
)

model = LogisticRegression(solver="liblinear", C=95, penalty="l1")
model.fit(x_training, y_training)

score = model.score(x_test, y_test)
# print("Acurácia", score)

prediction = model.predict_proba(x_test)
probs = prediction[:, 1]
# print(probs)

fpr, tpr, thresholds = roc_curve(y_test, probs)
# print("FPR:", fpr)  # false positives rate
# print("TPR:", tpr)  # true positives rate
# print("THRESHOLDS:", thresholds)

plt.scatter(fpr, tpr)
plt.show()

print(
    roc_auc_score(y_test, probs)
)  # área abaixo da curva, quanto mais prox de 1 melhor
