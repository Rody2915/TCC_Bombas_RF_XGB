import pandas as pd
import numpy as np
from scipy.stats import kurtosis

arquivo = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration\Motor-2\75\bearing bpfi 1/Vibration_Motor-2_75_time-bearing bpfi 1-ch4.csv"

df = pd.read_csv(arquivo)

resultados = []

for coluna in df.columns[1:]:

    sinal = df[coluna]

    rms = np.sqrt(np.mean(sinal**2))

    pico = np.max(np.abs(sinal))

    media = np.mean(sinal)

    desvio = np.std(sinal)

    curtose = kurtosis(sinal)

    fator_crista = pico / rms



    resultados.append({
        "Medição": coluna,

        "RMS": rms,

        "Pico": pico,

        "Média": media,

        "Desvio": desvio,

        "Curtose": curtose,

        "fator_crista": fator_crista

    })

print("Mínimo =", sinal.min())

print("Máximo =", sinal.max())

sinal = df["9"]
print(sinal.describe())



print("\n===== RESULTADOS =====\n")

base_features = pd.DataFrame(resultados)


print(base_features)