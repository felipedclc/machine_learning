import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# APENAS NÚMEROS

pd.set_option("display.width", 320)

columns = [
    "preg",
    "plas",
    "pres",
    "skin",
    "test",
    "mass",
    "pedi",
    "age",
    "class",
]

data = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\diabetes.csv"
)


# obtendo a correlação dos dados
# print(data.corr(method="pearson"))  # método de Pearson = mais utilizado


# MAPA DE CALOR ENTRE AS CORRELAÇÕES
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr())
plt.show()
