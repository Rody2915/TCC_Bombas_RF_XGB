import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Métricas e Sinais", page_icon="📈", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_VISUALIZACOES = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "visualizacoes"))
DIR_DADOS = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "dados_processados"))

st.title("📈 Análise Dinâmica do Sinal e Desempenho da IA")
st.markdown("---")

# ==============================================================================
# SEÇÃO 1: GRÁFICOS DINÂMICOS DO ARQUIVO ANEXADO
# ==============================================================================
st.header("📊 Diagnóstico Físico do Ensaio Atual")

if "sinal_bruto" in st.session_state:
    st.success(
        f"🔍 Exibindo gráficos do sinal dinâmico avaliado como: "
        f"**{st.session_state['diagnostico_atual']}** (Severidade: {st.session_state['severidade_atual']})"
    )
    
    sinal_bruto = st.session_state["sinal_bruto"]
    features = st.session_state["features_atuais"]
    fs = 20000

    # Aplica a mesma centralização e corte via MAD usados no treinamento para visualização fiel
    mediana_sinal = np.median(sinal_bruto)
    sinal_centralizado = sinal_bruto - mediana_sinal
    mad = np.median(np.abs(sinal_centralizado))
    if mad > 0:
        limite = (3 * mad) * 1.4826
        sinal_filtrado = np.clip(sinal_centralizado, -limite, limite)
    else:
        sinal_filtrado = sinal_centralizado

    # Cálculos dinâmicos da FFT (entre 5 Hz e 2000 Hz, idêntico ao treinamento)
    tempo = np.linspace(0, len(sinal_filtrado) / fs, len(sinal_filtrado))
    fft = np.fft.fft(sinal_filtrado)
    frequencias = np.fft.fftfreq(len(sinal_filtrado), d=1/fs)
    amplitudes = np.abs(fft)

    metade = len(amplitudes) // 2
    frequencias = frequencias[:metade]
    amplitudes = amplitudes[:metade]
    mascara_banda = (frequencias > 5) & (frequencias <= 2000)

    col_sinal1, col_sinal2 = st.columns(2)

    with col_sinal1:
        st.markdown("### 🕒 Forma de Onda no Domínio do Tempo")
        fig_tempo, ax_t = plt.subplots(figsize=(10, 4.5))
        ax_t.plot(tempo[:2000], sinal_filtrado[:2000], color="#1f77b4", alpha=0.85)
        ax_t.set_title("Sinal de Vibração Tratado via MAD (Primeiros 0,1 s)")
        ax_t.set_xlabel("Tempo (s)")
        ax_t.set_ylabel("Amplitude de Aceleração (g)")
        ax_t.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig_tempo)
        plt.close(fig_tempo)

    with col_sinal2:
        st.markdown("### 🌌 Espectro de Frequência (FFT)")
        fig_freq, ax_f = plt.subplots(figsize=(10, 4.5))
        ax_f.plot(frequencias[mascara_banda], amplitudes[mascara_banda], color="#ff7f0e")
        ax_f.set_title("Espectro de Amplitude Harmônica (5 Hz a 2000 Hz)")
        ax_f.set_xlabel("Frequência (Hz)")
        ax_f.set_ylabel("Amplitude")
        ax_f.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig_freq)
        plt.close(fig_freq)

    st.markdown("### 📋 Vetor de 36 Indicadores Calculado para este Arquivo")
    st.dataframe(features, use_container_width=True)

else:
    st.info(
        "💡 **Nenhum arquivo foi anexado ainda.** Vá até a página **1_Analise** no menu lateral, "
        "faça o upload de um arquivo CSV e clique em processar para liberar os gráficos em tempo real desta seção."
    )

st.markdown("---")

# ==============================================================================
# SEÇÃO 2: ESPAÇO GLOBAL REAL DE INDICADORES (DADOS EXPERIMENTAIS DO TCC)
# ==============================================================================
st.header("🔮 Espaço Global de Indicadores: Dados Experimentais Reais")
st.markdown(
    "Esta visualização interativa mapeia as **amostras reais processadas do conjunto motobomba**, "
    "permitindo inspecionar como os indicadores físicos no domínio do tempo e da frequência separam as condições operacionais."
)

col_filt1, col_filt2, col_filt3 = st.columns(3)
with col_filt1:
    pot_selecionada = st.selectbox("Velocidade de Operação (Base Real)", ["50%", "75%", "100%"], index=2)
with col_filt2:
    eixo_x_escolhido = st.selectbox(
        "Indicador no Eixo X",
        ["Desvio_Padrao_Espectral", "Variancia_Espectral", "Curtose", "Centroide_Espectral", "Frequencia_Media_4a_Ordem"],
        index=0
    )
with col_filt3:
    eixo_y_escolhido = st.selectbox(
        "Indicador no Eixo Y",
        ["RMS", "Curtose", "Fator_Crista", "Media_Espectral", "Frequencia_RMS"],
        index=0
    )

caminho_base_real = os.path.join(DIR_DADOS, f"base_features_motor2_{pot_selecionada.replace('%', '')}pct.xlsx")

if os.path.exists(caminho_base_real):
    @st.cache_data
    def carregar_base_processada(caminho):
        return pd.read_excel(caminho)

    df_real = carregar_base_processada(caminho_base_real)

    # Limita a visualização a até 25 amostras por classe preservando a coluna "Classe"
    if len(df_real) > 600:
        df_plot = (
            df_real.sample(frac=1, random_state=42)
            .groupby("Classe", as_index=False)
            .head(25)
            .reset_index(drop=True)
        )
    else:
        df_plot = df_real


    fig_scatter = px.scatter(
        df_plot,
        x=eixo_x_escolhido,
        y=eixo_y_escolhido,
        color="Classe",
        hover_data=["Canal_Sensor", "RMS", "Curtose", "Freq_Pico_1"],
        title=f"Distribuição Experimental Real ({pot_selecionada} da Capacidade) — {eixo_x_escolhido} vs. {eixo_y_escolhido}",
        template="plotly_white"
    )

    fig_scatter.update_traces(
        marker=dict(size=8, opacity=0.82, line=dict(width=0.6, color="Black")),
        selector=dict(mode="markers")
    )
    fig_scatter.update_layout(
        legend_title_text="Condição / Falha",
        margin=dict(l=40, r=40, t=55, b=40),
        hovermode="closest"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.warning(f"Base processada `{os.path.basename(caminho_base_real)}` não encontrada na pasta `dados_processados/`.")

st.markdown("---")

# ==============================================================================
# SEÇÃO 3: AUDITORIA DE DESEMPENHO DOS MODELOS (RF vs. XGBOOST)
# ==============================================================================
st.header("🏆 Validação Experimental: Random Forest vs. XGBoost")

aba_metricas, aba_importancia, aba_matrizes = st.tabs([
    "📊 Comparativo de Métricas",
    "🎯 Importância das Variáveis",
    "🧩 Matrizes de Confusão"
])

with aba_metricas:
    pot_m = st.radio("Selecione a velocidade para visualizar as métricas:", ["50%", "75%", "100%"], horizontal=True, key="rad_m")
    img_m = os.path.join(DIR_VISUALIZACOES, "metricas", f"comparativo_metricas_{pot_m.replace('%', '')}pct.png")
    if os.path.exists(img_m):
        st.image(img_m, use_container_width=True)
    else:
        st.info(f"Gráfico `{os.path.basename(img_m)}` não encontrado em `visualizacoes/metricas/`.")

with aba_importancia:
    pot_i = st.radio("Selecione a velocidade para visualizar o Top 15:", ["50%", "75%", "100%"], horizontal=True, key="rad_i")
    img_i = os.path.join(DIR_VISUALIZACOES, "importancia_variaveis", f"comparativo_importancia_rf_vs_xgb_{pot_i.replace('%', '')}pct.png")
    if os.path.exists(img_i):
        st.image(img_i, use_container_width=True)
    else:
        st.info(f"Gráfico `{os.path.basename(img_i)}` não encontrado em `visualizacoes/importancia_variaveis/`.")

with aba_matrizes:
    col_mz1, col_mz2 = st.columns(2)
    with col_mz1:
        alg_mz = st.selectbox("Algoritmo", ["Random Forest", "XGBoost"], key="sel_alg_mz")
    with col_mz2:
        pot_mz = st.selectbox("Velocidade", ["50%", "75%", "100%"], key="sel_pot_mz")
    
    cod_mz = "rf" if alg_mz == "Random Forest" else "xgb"
    img_mz = os.path.join(DIR_VISUALIZACOES, "matrizes_confusao", f"matriz_confusao_{cod_mz}_{pot_mz.replace('%', '')}pct.png")
    if os.path.exists(img_mz):
        st.image(img_mz, use_container_width=True)
    else:
        st.info(f"Matriz `{os.path.basename(img_mz)}` não encontrada em `visualizacoes/matrizes_confusao/`.")
