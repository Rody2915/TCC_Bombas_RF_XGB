import pandas as pd
import matplotlib.pyplot as plt

arquivo = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration\Motor-2\75\bearing bpfi 1/Vibration_Motor-2_75_time-bearing bpfi 1-ch4.csv"


df = pd.read_csv(arquivo)

sinal = df["9"]

indice_max = sinal.idxmax()

inicio = indice_max - 100

fim = indice_max + 100

plt.figure(figsize=(12,5))

plt.plot(
    sinal.iloc[inicio:fim]
)

plt.title(
    "Evento impulsivo - Bearing BPFI 1 - Canal 4"
)

plt.xlabel("Amostras")

plt.ylabel("Aceleração (g)")

plt.grid()

plt.show()