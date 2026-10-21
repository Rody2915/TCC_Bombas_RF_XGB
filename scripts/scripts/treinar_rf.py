import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

arquivo = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\base_features_geral.xlsx"

df = pd.read_excel(arquivo)

X = df[
    [
        "RMS",
        "Pico",
        "Media",
        "Desvio",
        "Curtose",
        "Fator_Crista"
    ]
]

y = df["Classe"]

X_train, X_test, y_train, y_test =train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

modelo.fit(
    X_train,
    y_train
)

print()

print("Modelo treinado com sucesso!")

predicoes = modelo.predict(X_test)

print()

print("Primeiras previsões:")

print(predicoes[:10])

from sklearn.metrics import accuracy_score

acuracia = accuracy_score(
    y_test,
    predicoes
)

print()

print("Acurácia:")

print(acuracia)

from sklearn.metrics import classification_report

print()

print(
    classification_report(
        y_test,
        predicoes
    )
)

matriz = confusion_matrix(
    y_test,
    predicoes
)



print()


import pandas as pd

importancias = pd.DataFrame(
    {
        "Variavel": X.columns,
        "Importancia": modelo.feature_importances_
    }
)

importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

print()

print("IMPORTÂNCIA DAS VARIÁVEIS")

print(importancias)



plt.figure(figsize=(15, 10))

sns.heatmap(
    matriz,
    cmap="Blues"
)

plt.title(
    "Matriz de Confusão - Random Forest"
)

plt.xlabel("Classe Prevista")

plt.ylabel("Classe Real")

plt.show()

importancias.to_excel(
    "importancia_variaveis.xlsx",
    index=False
)

print("\nImportância salva!")

joblib.dump(
    modelo,
    "modelo_rf_v1.pkl"
)

print(
    "\nModelo salvo com sucesso!"
)

