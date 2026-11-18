import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis


def extrair_features(arquivo, classe):

    df = pd.read_csv(arquivo)

    resultados = []

    for coluna in df.columns[1:]:

        sinal = df[coluna]

        fs = 20000
        
        # FFT
        fft = np.fft.fft(sinal)
        
        #   Frequências correspondentes
        frequencias = np.fft.fftfreq(
            len(sinal),
            d=1/fs
        )
        # Amplitudes
        amplitudes = np.abs(fft)

        # Manter apenas a metade positiva do espectro
        metade = len(amplitudes) // 2

        frequencias = frequencias[:metade]
        amplitudes = amplitudes[:metade]

        # Ignorar componente DC (0 Hz)

        mascara = frequencias <= 2000

        frequencias_baixas = frequencias[mascara]

        amplitudes_baixas = amplitudes[mascara]

        from scipy.signal import find_peaks

        picos, _ = find_peaks(
            amplitudes_baixas,
            prominence=500,
            distance=1000
        )

    # amplitudes dos picos físicos
        amplitudes_picos = amplitudes_baixas[picos]
    
        ordem = np.argsort(
            amplitudes_picos
        )[::-1]
    
        # garantir pelo menos 3 picos
        if len(ordem) >= 3:
    
            freq1 = frequencias_baixas[picos[ordem[0]]]
            amp1  = amplitudes_baixas[picos[ordem[0]]]
    
            freq2 = frequencias_baixas[picos[ordem[1]]]
            amp2  = amplitudes_baixas[picos[ordem[1]]]
    
            freq3 = frequencias_baixas[picos[ordem[2]]]
            amp3  = amplitudes_baixas[picos[ordem[2]]]
    
        else:
    
            freq1 = amp1 = np.nan
            freq2 = amp2 = np.nan
            freq3 = amp3 = np.nan        
        
        ## Vibration
        rms = np.sqrt(np.mean(sinal ** 2))

        pico = np.max(np.abs(sinal))

        # media = np.mean(sinal)

        desvio = np.std(sinal)

        curtose_val = kurtosis(
            sinal,
            fisher=False
        )

        fator_crista = pico / rms

        resultados.append({

        
            "RMS": rms,

            "Pico": pico,

          #  "Media": media,

            "Desvio": desvio,

            "Curtose": curtose_val,

            "Fator_Crista": fator_crista,

            "Freq_Pico_1": freq1,
            "Amp_Pico_1": amp1,
    
            "Freq_Pico_2": freq2,
            "Amp_Pico_2": amp2,
    
            "Freq_Pico_3": freq3,
            "Amp_Pico_3": amp3,            

            "Classe": classe

        })

    return pd.DataFrame(resultados)


caminho_principal = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration\Motor-2\75"

todas_as_classes = []

for pasta in os.listdir(caminho_principal):

    caminho_pasta = os.path.join(
        caminho_principal,
        pasta
    )

    if os.path.isdir(caminho_pasta):

        print(f"\nProcessando: {pasta}")

        for arquivo in os.listdir(caminho_pasta):

            if arquivo.endswith("-ch4.csv"):

                caminho_csv = os.path.join(
                    caminho_pasta,
                    arquivo
                )

                resultado = extrair_features(
                    caminho_csv,
                    pasta
                )

                print(
                    f"Amostras encontradas: {len(resultado)}"
                )

                todas_as_classes.append(
                    resultado
                )

base_final = pd.concat(
    todas_as_classes,
    ignore_index=True
)

print("\n================================")
print("BASE FINAL CRIADA")
print("================================")

print(
    f"Total de amostras: {len(base_final)}"
)

print("\nPrimeiras linhas:\n")

print(base_final.head())

base_final.to_excel(
    "base_features_completa.xlsx",
    index=False
)

print(
    "\nArquivo base_features_completa.xlsx salvo com sucesso!"
)

print()

print(
    base_final["Classe"].value_counts()
)

##############################################################


