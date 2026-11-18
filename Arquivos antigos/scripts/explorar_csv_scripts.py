import pandas as pd

arquivo = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration\Motor-2\75\bearing bpfi 1/Vibration_Motor-2_75_time-bearing bpfi 1-ch4.csv"

df = pd.read_csv(arquivo)

print(df.head())

print("\n")

print("Número de linhas:", len(df))

print("Número de colunas:", len(df.columns))

print("\nColunas:")

print(df.columns)