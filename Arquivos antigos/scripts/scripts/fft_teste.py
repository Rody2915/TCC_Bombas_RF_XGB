import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


arquivo = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration\Motor-2\75\bearing bpfi 1\Vibration_Motor-2_75_time-bearing bpfi 1-ch4.csv"

df = pd.read_csv(arquivo)

# Primeira medição
sinal = df["0"]

# Frequência de amostragem do dataset
fs = 20000

# FFT
fft = np.fft.fft(sinal)

# Frequências correspondentes
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
indice_pico = np.argmax(amplitudes[1:]) + 1

freq_dominante = frequencias[indice_pico]
amp_dominante = amplitudes[indice_pico]

mascara = frequencias <= 2000

frequencias_baixas = frequencias[mascara]

amplitudes_baixas = amplitudes[mascara]

from scipy.signal import find_peaks

picos, _ = find_peaks(
    amplitudes_baixas,
    prominence=500,
    distance=1000
)


plt.figure(figsize=(12,6))

plt.plot(
    frequencias_baixas,
    amplitudes_baixas
)

plt.plot(
    frequencias_baixas[picos],
    amplitudes_baixas[picos],
    "ro"
)

plt.title(
    "FFT com picos detectados"
)

plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")

plt.grid()

plt.show()


print("\nPICOS ENCONTRADOS\n")

for pico in picos:

    print(
        f"Freq: {frequencias_baixas[pico]:.2f} Hz "
        f"{amplitudes_baixas[pico]:.2f}"
    )

indice_pico = np.argmax(amplitudes_baixas[1:]) + 1

freq_dominante = frequencias_baixas[indice_pico]

amp_dominante = amplitudes_baixas[indice_pico]

print()

# Top 10 picos do espectro

indices_maiores = np.argsort(
    amplitudes_baixas
)[-10:]

print("\nTOP 10 PICOS\n")

for i in reversed(indices_maiores):

    print(
        f"Freq: {frequencias_baixas[i]:.2f} Hz "
        f"| Amp: {amplitudes_baixas[i]:.2f}"
    )

print("Frequência dominante:", freq_dominante, "Hz")

print("Amplitude dominante:", amp_dominante)

# Gráfico
plt.figure(figsize=(12,6))

plt.plot(
    frequencias_baixas,
    amplitudes_baixas
)

plt.title(
    "FFT - bearing bpfi 1 - Canal 4 - Medição 0"
)

plt.xlabel("Frequência (Hz)")

plt.ylabel("Amplitude")

plt.grid()

plt.show()