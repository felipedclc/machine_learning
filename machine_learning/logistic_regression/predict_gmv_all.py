from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot

import pandas as pd

file = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\all_emissions.csv"
)

file.dropna(inplace=True)

print(file.shape)

print(file.head(100))

print("-----------------------------------------------------------------")

print(file.describe())

file.plot(
    kind="line", subplots=True, layout=[12, 2], sharex=False, sharey=False
)

pyplot.show()

array = file.values
X = array[:, 0:15]
y = array[:, 15]
print(y)
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

models = []
models.append(
    ("LR", LogisticRegression(solver="liblinear", multi_class="ovr"))
)
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
# models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))


# Here is where the errors are shown


results = []
names = []
print("---- X - Train ---------")
print(X_train)
print(Y_train)

for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    print(name)
    cv_results = cross_val_score(
        model,
        X_train,
        Y_train,
        cv=kfold.get_n_splits(X_train, Y_train),
        scoring="accuracy",
    )
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
