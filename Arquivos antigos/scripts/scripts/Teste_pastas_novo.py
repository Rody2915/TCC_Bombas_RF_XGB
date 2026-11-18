import pandas as pd
import numpy as np
from scipy.stats import kurtosis





def extrair_features(arquivo, classe):

    df = pd.read_csv(arquivo)

    resultados = []

    for coluna in df.columns[1:]:

        sinal = df[coluna]

        rms = np.sqrt(np.mean(sinal ** 2))

        pico = np.max(np.abs(sinal))

        media = np.mean(sinal)

        desvio = np.std(sinal)

        curtose_val = kurtosis(
            sinal,
            fisher=False
        )

        fator_crista = pico / rms

        resultados.append({

            "Amostra": int(coluna),

            "RMS": rms,

            "Pico": pico,

            "Media": media,

            "Desvio": desvio,

            "Curtose": curtose_val,

            "Fator_Crista": fator_crista,

            "Classe": classe

        })

    return pd.DataFrame(resultados)

arquivo = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration\Motor-2\75\healthy 1\Vibration_Motor-2_75_time-healthy 1-ch4.csv"

resultado = extrair_features(
    arquivo,
    "healthy 1"
)

print(resultado.head())

print()

print("Quantidade de amostras:")

print(len(resultado))

arquivo_bpfi = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration\Motor-2\75\bearing bpfi 1\Vibration_Motor-2_75_time-bearing bpfi 1-ch4.csv"

resultado_bpfi = extrair_features(
    arquivo_bpfi,
    "bearing bpfi 1"
)

print()

print("Quantidade de amostras BPFI:")

print(len(resultado_bpfi))