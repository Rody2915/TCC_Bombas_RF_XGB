import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ==============================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS DO SEU PC
# ==============================================================================
diretorio_dataset = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration"
diretorio_modelo_saida = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\modelo"
diretorio_visualizacoes = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\visualizacoes"
diretorio_bases = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_processados"

os.makedirs(diretorio_modelo_saida, exist_ok=True)
os.makedirs(diretorio_visualizacoes, exist_ok=True)
os.makedirs(diretorio_bases, exist_ok=True)

# ==============================================================================
# 2. EXTRAÇÃO DE FEATURES (NOMES DESCRITIVOS NO TEMPO E NA FREQUÊNCIA)
# ==============================================================================
def extrair_features_matriz(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    resultados = []
    fs = 20000
    eps = 1e-12  # Proteção numérica contra divisão por zero

    for coluna in df.columns[1:]:
        sinal_bruto = df[coluna].values

        # --- TRATAMENTO ESTATÍSTICO ROBUSTO (MAD & Z-SCORE MODIFICADO) ---
        mediana_sinal = np.median(sinal_bruto)
        sinal_centralizado = sinal_bruto - mediana_sinal
        mad = np.median(np.abs(sinal_centralizado))

        if mad > 0:
            limite_estatistico = (3 * mad)*1.4826
            sinal = np.clip(sinal_centralizado, -limite_estatistico, limite_estatistico)
        else:
            sinal = sinal_centralizado

        N = len(sinal)

        # ======================================================================
        # DOMÍNIO DO TEMPO (16 Indicadores Físicos e Estatísticos)
        # ======================================================================
        media = np.mean(sinal)
        rms = np.sqrt(np.mean(sinal ** 2))
        amplitude_raiz_quadrada = (np.mean(np.sqrt(np.abs(sinal)))) ** 2
        media_absoluta = np.mean(np.abs(sinal))
        momento_central_3 = np.mean((sinal - media) ** 3)
        momento_central_4 = np.mean((sinal - media) ** 4)
        valor_maximo = np.max(sinal)
        valor_minimo = np.min(sinal)
        pico_a_pico = valor_maximo - valor_minimo
        variancia = np.sum((sinal - media) ** 2) / (N - 1)
        fator_forma = rms / (media_absoluta + eps)
        fator_crista = valor_maximo / (rms + eps)
        pico_maximo = valor_maximo
        amplitude_media_absoluta = media_absoluta
        assimetria = momento_central_3 / ((rms ** 3) + eps)
        curtose = momento_central_4 / ((rms ** 4) + eps)

        # ======================================================================
        # PROCESSAMENTO FFT (Domínio da Frequência)
        # ======================================================================
        fft = np.fft.fft(sinal)
        frequencias = np.fft.fftfreq(N, d=1/fs)
        amplitudes = np.abs(fft)

        metade = len(amplitudes) // 2
        frequencias = frequencias[:metade]
        amplitudes = amplitudes[:metade]

        # Filtra componente contínua (< 5 Hz) e limita até 2000 Hz
        mascara = (frequencias > 5) & (frequencias <= 2000)
        f_k = frequencias[mascara]
        y_k = amplitudes[mascara]
        K = len(y_k)

        # --- TRÊS MAIORES PICOS ESPECTRAIS ---
        picos, _ = find_peaks(y_k, distance=15)
        amplitudes_picos = y_k[picos]
        ordem = np.argsort(amplitudes_picos)[::-1]

        freq_pico_1 = amp_pico_1 = freq_pico_2 = amp_pico_2 = freq_pico_3 = amp_pico_3 = 0.0
        if len(ordem) >= 1:
            freq_pico_1 = float(f_k[picos[ordem[0]]])
            amp_pico_1 = float(y_k[picos[ordem[0]]])
        if len(ordem) >= 2:
            freq_pico_2 = float(f_k[picos[ordem[1]]])
            amp_pico_2 = float(y_k[picos[ordem[1]]])
        if len(ordem) >= 3:
            freq_pico_3 = float(f_k[picos[ordem[2]]])
            amp_pico_3 = float(y_k[picos[ordem[2]]])

        # ======================================================================
        # DOMÍNIO DA FREQUÊNCIA (13 Indicadores Espectrais)
        # ======================================================================
        soma_y = np.sum(y_k) + eps

        media_espectral = np.mean(y_k)
        variancia_espectral = np.sqrt(np.sum((y_k - media_espectral) ** 2) / (K - 1))
        assimetria_espectral = np.sum((y_k - media_espectral) ** 3) / (K * (np.sqrt(variancia_espectral) ** 3) + eps)
        curtose_espectral = np.sum((y_k - media_espectral) ** 4) / (K * (variancia_espectral ** 4) + eps)

        centroide_espectral = np.sum(f_k * y_k) / soma_y
        desvio_padrao_espectral = np.sqrt(np.sum(((f_k - centroide_espectral) ** 2) * y_k) / K)

        soma_f2_y = np.sum((f_k ** 2) * y_k)
        soma_f4_y = np.sum((f_k ** 4) * y_k)

        frequencia_rms = np.sqrt(soma_f2_y / soma_y)
        frequencia_media_quarta_ordem = np.sqrt(soma_f4_y / (soma_f2_y + eps))
        fator_estabilizacao_espectral = soma_f2_y / (np.sqrt(soma_y * soma_f2_y) + eps)
        coeficiente_variacao_espectral = centroide_espectral / (desvio_padrao_espectral + eps)
        assimetria_frequencia = np.sum(((f_k - centroide_espectral) ** 3) * y_k) / (K * (centroide_espectral ** 3) + eps)
        curtose_frequencia = np.sum(((f_k - centroide_espectral) ** 4) * y_k) / (K * (centroide_espectral ** 4) + eps)
        dispersao_raiz_frequencia = np.sum(np.sqrt(np.abs(f_k - centroide_espectral)) * y_k) / (K * np.sqrt(centroide_espectral) + eps)

        resultados.append({
            "Media": media,
            "RMS": rms,
            "Amplitude_Raiz_Quadrada": amplitude_raiz_quadrada,
            "Media_Absoluta": media_absoluta,
            "Momento_Central_3": momento_central_3,
            "Momento_Central_4": momento_central_4,
            "Valor_Maximo": valor_maximo,
            "Valor_Minimo": valor_minimo,
            "Pico_a_Pico": pico_a_pico,
            "Variancia": variancia,
            "Fator_Forma": fator_forma,
            "Fator_Crista": fator_crista,
            "Pico_Maximo": pico_maximo,
            "Amplitude_Media_Absoluta": amplitude_media_absoluta,
            "Assimetria": assimetria,
            "Curtose": curtose,
            "Media_Espectral": media_espectral,
            "Variancia_Espectral": variancia_espectral,
            "Assimetria_Espectral": assimetria_espectral,
            "Curtose_Espectral": curtose_espectral,
            "Desvio_Padrao_Espectral": desvio_padrao_espectral,
            "Centroide_Espectral": centroide_espectral,
            "Frequencia_RMS": frequencia_rms,
            "Frequencia_Media_4a_Ordem": frequencia_media_quarta_ordem,
            "Fator_Estabilizacao_Espectral": fator_estabilizacao_espectral,
            "Coeficiente_Variacao_Espectral": coeficiente_variacao_espectral,
            "Assimetria_Frequencia": assimetria_frequencia,
            "Curtose_Frequencia": curtose_frequencia,
            "Dispersao_Raiz_Frequencia": dispersao_raiz_frequencia,
            "Freq_Pico_1": freq_pico_1,
            "Amp_Pico_1": amp_pico_1,
            "Freq_Pico_2": freq_pico_2,
            "Amp_Pico_2": amp_pico_2,
            "Freq_Pico_3": freq_pico_3,
            "Amp_Pico_3": amp_pico_3
        })

    return pd.DataFrame(resultados)


# ==============================================================================
# 3. MAPEAMENTO DE PASTAS E CANAIS
# ==============================================================================
MAPEAMENTO_PASTAS = {
    "bearing bpfi 1": "bearing bpfi 1", "bearing bpfi 2": "bearing bpfi 2", "bearing bpfi 3": "bearing bpfi 3",
    "bearing bpfo 1": "bearing bpfo 1", "bearing bpfo 2": "bearing bpfo 2", "bearing bpfo 3": "bearing bpfo 3",
    "bearing bsf": "bearing bsf", "bearing contaminated": "bearing contaminated", "broken rotor bar": "broken rotor bar",
    "bearing pump 1": "bearing pump 1", "bearing pump 2": "bearing pump 2", "bearing pump 3": "bearing pump 3",
    "healthy 1": "healthy", "healthy 2": "healthy", "healthy 3": "healthy", "healthy noise": "healthy noise",
    "impeller 1": "impeller 1", "impeller 2": "impeller 2", "impeller 3": "impeller 3",
    "loose foot motor": "loose foot motor", "loose foot pump": "loose foot pump", "new motor": "new motor",
    "soft foot 1": "soft foot 1", "soft foot 2": "soft foot 2",
    "stator short 1": "stator short 1", "stator short 2": "stator short 2"
}

MAPA_NUMERICO_CANAIS = {"-ch1.csv": 1, "-ch2.csv": 2, "-ch3.csv": 3, "-ch4.csv": 4, "-ch5.csv": 5}
velocidades = ["50", "75", "100"]
resumo_comparativo = []

# Ordem fixa das 36 colunas de entrada com nomes descritivos
COLUNAS_FEATURES = [
    "Canal_Sensor",
    "Media", "RMS", "Amplitude_Raiz_Quadrada", "Media_Absoluta",
    "Momento_Central_3", "Momento_Central_4", "Valor_Maximo", "Valor_Minimo",
    "Pico_a_Pico", "Variancia", "Fator_Forma", "Fator_Crista",
    "Pico_Maximo", "Amplitude_Media_Absoluta", "Assimetria", "Curtose",
    "Media_Espectral", "Variancia_Espectral", "Assimetria_Espectral", "Curtose_Espectral",
    "Desvio_Padrao_Espectral", "Centroide_Espectral", "Frequencia_RMS", "Frequencia_Media_4a_Ordem",
    "Fator_Estabilizacao_Espectral", "Coeficiente_Variacao_Espectral", "Assimetria_Frequencia",
    "Curtose_Frequencia", "Dispersao_Raiz_Frequencia",
    "Freq_Pico_1", "Amp_Pico_1", "Freq_Pico_2", "Amp_Pico_2", "Freq_Pico_3", "Amp_Pico_3"
]

# ==============================================================================
# 4. LAÇO PRINCIPAL POR POTÊNCIA E OTIMIZAÇÃO
# ==============================================================================

for vel in velocidades:
    print(f"\n{'='*70}")
    print(f"🚀 PROCESSANDO REGIME OPERACIONAL: {vel}% DA CAPACIDADE")
    print(f"{'='*70}")

    todas_as_classes = []
    caminho_potencia = os.path.join(diretorio_dataset, "Motor-2", vel)

    for subpasta, label_classe in MAPEAMENTO_PASTAS.items():
        caminho_completo = os.path.join(caminho_potencia, subpasta)

        if os.path.exists(caminho_completo):
            print(f"📁 [{vel}%] Extraindo: {label_classe}...")
            todos_arquivos = os.listdir(caminho_completo)

            for sufixo_ch, id_canal in MAPA_NUMERICO_CANAIS.items():
                arquivos_canal = [f for f in todos_arquivos if f.endswith(sufixo_ch)]

                for arquivo_csv in arquivos_canal:
                    caminho_arquivo = os.path.join(caminho_completo, arquivo_csv)
                    try:
                        df_features = extrair_features_matriz(caminho_arquivo)
                        df_features["Canal_Sensor"] = id_canal
                        df_features["Potencia_Pct"] = int(vel)
                        df_features["Classe"] = label_classe
                        todas_as_classes.append(df_features)
                    except Exception as e:
                        print(f"   ⚠️ Erro em {arquivo_csv}: {e}")
        else:
            print(f"   ⚠️ Pasta não encontrada: {caminho_completo}")

    if not todas_as_classes:
        print(f"❌ Nenhum dado encontrado para {vel}%. Pulando...")
        continue

    base_potencia = pd.concat(todas_as_classes, ignore_index=True)
    nome_planilha = os.path.join(diretorio_bases, f"base_features_motor2_{vel}pct.xlsx")
    base_potencia.to_excel(nome_planilha, index=False)
    print(f"\n✅ Base salva ({len(base_potencia)} amostras): {nome_planilha}")

    # Divisão 75% Treino e 25% Teste
    X = base_potencia[COLUNAS_FEATURES]
    y = base_potencia["Classe"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train)
    y_test_num = le.transform(y_test)

    # Validação Cruzada Estratificada k = 10
    cv_10_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # --------------------------------------------------------------------------
    # A. GRID SEARCH: RANDOM FOREST
    # --------------------------------------------------------------------------
    print(f"\n🌲 Executando Grid Search (k=10) para Random Forest ({vel}%)...")
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 25],
        'min_samples_split': [2, 5]
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=cv_10_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_rf.fit(X_train, y_train_num)
    melhor_rf = grid_rf.best_estimator_
    acc_rf = accuracy_score(y_test_num, melhor_rf.predict(X_test))

    print(f"✅ RF ({vel}%) -> CV: {grid_rf.best_score_*100:.2f}% | Teste: {acc_rf*100:.2f}%")
    print(f"   Melhores Params RF: {grid_rf.best_params_}")

    # --------------------------------------------------------------------------
    # B. GRID SEARCH: XGBOOST
    # --------------------------------------------------------------------------
    print(f"\n🚀 Executando Grid Search (k=10) para XGBoost ({vel}%)...")
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    grid_xgb = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='mlogloss'),
        param_grid_xgb,
        cv=cv_10_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_xgb.fit(X_train, y_train_num)
    melhor_xgb = grid_xgb.best_estimator_
    acc_xgb = accuracy_score(y_test_num, melhor_xgb.predict(X_test))

    print(f"✅ XGB ({vel}%) -> CV: {grid_xgb.best_score_*100:.2f}% | Teste: {acc_xgb*100:.2f}%")
    print(f"   Melhores Params XGB: {grid_xgb.best_params_}")
    # Convergência de árvores (100, 200 e 300)
    resultados_cv_rf = pd.DataFrame(grid_rf.cv_results_)
    comparativo_arvores_rf = resultados_cv_rf.groupby('param_n_estimators').agg(
        Acuracia_Media_Pct=('mean_test_score', lambda x: round(x.mean() * 100, 2)),
        Acuracia_Maxima_Pct=('mean_test_score', lambda x: round(x.max() * 100, 2)),
        Tempo_Medio_Seg=('mean_fit_time', lambda x: round(x.mean(), 3))
    ).reset_index()
    comparativo_arvores_rf.to_excel(
        os.path.join(diretorio_visualizacoes, f"convergencia_arvores_rf_{vel}pct.xlsx"),
        index=False
    )

    resultados_cv_xgb = pd.DataFrame(grid_xgb.cv_results_)
    comparativo_arvores_xgb = resultados_cv_xgb.groupby('param_n_estimators').agg(
        Acuracia_Media_Pct=('mean_test_score', lambda x: round(x.mean() * 100, 2)),
        Acuracia_Maxima_Pct=('mean_test_score', lambda x: round(x.max() * 100, 2)),
        Tempo_Medio_Seg=('mean_fit_time', lambda x: round(x.mean(), 3))
    ).reset_index()
    comparativo_arvores_xgb.to_excel(
        os.path.join(diretorio_visualizacoes, f"convergencia_arvores_xgb_{vel}pct.xlsx"),
        index=False
    )

    # --------------------------------------------------------------------------
    # C. REGISTRO COMPARATIVO, GRÁFICOS E EXPORTAÇÃO (.PKL)
    # --------------------------------------------------------------------------
    resumo_comparativo.append({
        "Potencia (%)": vel,
        "Amostras Totais": len(base_potencia),
        "RF Acuracia CV (%)": round(grid_rf.best_score_ * 100, 2),
        "RF Acuracia Teste (%)": round(acc_rf * 100, 2),
        "RF Melhores Params": str(grid_rf.best_params_),
        "XGB Acuracia CV (%)": round(grid_xgb.best_score_ * 100, 2),
        "XGB Acuracia Teste (%)": round(acc_xgb * 100, 2),
        "XGB Melhores Params": str(grid_xgb.best_params_)
    })

    # Salva pacotes compactados com o modelo, o encoder e a lista exata de colunas
    pacote_rf = {"modelo": melhor_rf, "encoder": le, "colunas": COLUNAS_FEATURES}
    pacote_xgb = {"modelo": melhor_xgb, "encoder": le, "colunas": COLUNAS_FEATURES}

    joblib.dump(pacote_rf, os.path.join(diretorio_modelo_saida, f"modelo_rf_{vel}pct.pkl"), compress=3)
    joblib.dump(pacote_xgb, os.path.join(diretorio_modelo_saida, f"modelo_xgb_{vel}pct.pkl"), compress=3)


# Exporta tabela consolidada para o TCC
df_resumo = pd.DataFrame(resumo_comparativo)
caminho_resumo = os.path.join(diretorio_visualizacoes, "comparativo_final_rf_vs_xgb.xlsx")
df_resumo.to_excel(caminho_resumo, index=False)
print(f"\n🎉 PROCESSAMENTO CONCLUÍDO! Tabela comparativa salva em: {caminho_resumo}")
