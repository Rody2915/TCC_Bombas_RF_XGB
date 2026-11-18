import os

caminho_principal = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration\Motor-2\75"

for pasta in os.listdir(caminho_principal):

    caminho_pasta = os.path.join(
        caminho_principal,
        pasta
    )

    print("\nClasse encontrada:")

    print(pasta)

    for arquivo in os.listdir(caminho_pasta):

        if arquivo.endswith("-ch4.csv"):

            print("Canal 4:")

            print(arquivo)