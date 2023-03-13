import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# APENAS

pd.set_option("display.width", 320)

data = pd.read_csv(
    "C:\\Users\\felip\\OneDrive\\Área de Trabalho\\estudo-nest\\python\\ocr\\machine_learning\\data\\vw_dataset_for_ml_predict.csv"
)

# data["onfly_amount"].fillna(data["onfly_amount"].mean(), inplace=True)
print(data.dtypes)

# data["total_amount"] = data["total_amount"].astype(float)
# data["onfly_amount"] = data["onfly_amount"].astype(float)
# print(data.head(100))

# obtendo a correlação dos dados
# print(data.corr(method="pearson"))  # método de Pearson = mais utilizado


# MAPA DE CALOR ENTRE AS CORRELAÇÕES
plt.figure(figsize=(5, 5))
sns.heatmap(data.corr())
plt.show()
