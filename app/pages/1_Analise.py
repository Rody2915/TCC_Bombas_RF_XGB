import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Configuração da página interna
st.set_page_config(page_title="Análise de Vibração", page_icon="📊", layout="wide")

# Diretório base dos modelos salvos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "modelo"))

@st.cache_resource
def carregar_pacote_ia(algoritmo_codigo, potencia_str):
    nome_arquivo = f"modelo_{algoritmo_codigo}_{potencia_str}pct.pkl"
    caminho_completo = os.path.join(PASTA_MODELOS, nome_arquivo)
    if os.path.exists(caminho_completo):
        return joblib.load(caminho_completo), nome_arquivo
    return None, nome_arquivo

# ==============================================================================
# FUNÇÃO DE EXTRAÇÃO SINCRONIZADA COM O TREINAMENTO FINAL (MAD + 36 INDICADORES)
# ==============================================================================
def extrair_features_geral(uploaded_file, canal_alvo, id_canal_ia):
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    if canal_alvo not in df.columns:
        raise ValueError(f"A coluna '{canal_alvo}' não foi encontrada no arquivo enviado.")
        
    sinal_bruto = df[canal_alvo].values
    fs = 20000
    eps = 1e-12

    # 1. Tratamento Estatístico Robusto (MAD & Z-Score Modificado)
    mediana_sinal = np.median(sinal_bruto)
    sinal_centralizado = sinal_bruto - mediana_sinal
    mad = np.median(np.abs(sinal_centralizado))

    if mad > 0:
        limite_estatistico = (3 * mad) * 1.4826
        sinal = np.clip(sinal_centralizado, -limite_estatistico, limite_estatistico)
    else:
        sinal = sinal_centralizado

    N = len(sinal)

    # 2. Domínio do Tempo (16 Indicadores)
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

    # 3. Processamento FFT (5 Hz a 2000 Hz)
    fft = np.fft.fft(sinal)
    frequencias = np.fft.fftfreq(N, d=1/fs)
    amplitudes = np.abs(fft)

    metade = len(amplitudes) // 2
    frequencias = frequencias[:metade]
    amplitudes = amplitudes[:metade]

    mascara = (frequencias > 5) & (frequencias <= 2000)
    f_k = frequencias[mascara]
    y_k = amplitudes[mascara]
    K = len(y_k)

    # Três Maiores Picos Espectrais
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

    # 4. Domínio da Frequência (13 Indicadores Espectrais)
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

    return pd.DataFrame([{
        "Canal_Sensor": id_canal_ia,
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
    }])

# ==============================================================================
# DICIONÁRIO TÉCNICO DE FALHAS
# ==============================================================================
DICON_FALHAS = {
    "healthy": {"nome": "Condição Saudável", "desc": "Equipamento operando dentro dos padrões ideais de projeto.", "cor": "green"},
    "healthy noise": {"nome": "Condição Saudável (Com Ruído)", "desc": "Assinatura operacional aceitável com presença de ruído de fundo.", "cor": "green"},
    "new motor": {"nome": "Motor Novo (Amaciamento)", "desc": "Componente em conformação inicial sem defeitos estruturais.", "cor": "green"},
    "bearing bpfi": {"nome": "Falha no Rolamento (Pista Interna)", "desc": "Defeito concentrado na pista interna dos elementos rotativos.", "cor": "red"},
    "bearing bpfo": {"nome": "Falha no Rolamento (Pista Externa)", "desc": "Defeito concentrado na pista externa do rolamento.", "cor": "red"},
    "bearing bsf": {"nome": "Falha no Rolamento (Esfera)", "desc": "Defeito micro-localizado nos elementos rotantes (esferas).", "cor": "orange"},
    "bearing contaminated": {"nome": "Rolamento Contaminado", "desc": "Falta de lubrificação eficiente ou infiltração de abrasivos.", "cor": "orange"},
    "bearing pump": {"nome": "Falha Geral no Rolamento da Bomba", "desc": "Avaria identificada no conjunto de suporte do eixo hidráulico.", "cor": "red"},
    "broken rotor bar": {"nome": "Barra Quebrada do Rotor", "desc": "Falha mecânica na barra de indução do rotor.", "cor": "red"},
    "impeller": {"nome": "Dano no Impulsor da Bomba", "desc": "Desgaste ou quebra nas pás hidráulicas.", "cor": "red"},
    "loose foot motor": {"nome": "Pé Frouxo no Motor", "desc": "Frouxidão mecânica identificada na fixação da base do motor.", "cor": "orange"},
    "loose foot pump": {"nome": "Pé Frouxo na Bomba", "desc": "Instabilidade mecânica na fixação estrutural da voluta da bomba.", "cor": "orange"},
    "soft foot": {"nome": "Pé Manco (Soft Foot)", "desc": "Problema crítico de nivelamento ou distorção geométrica na base.", "cor": "orange"},
    "stator short": {"nome": "Curto-Circuito no Estator", "desc": "Falha severa de isolamento nas bobinas elétricas do estator.", "cor": "red"}
}

def tratar_classe_label(classe_bruta):
    classe_limpa = str(classe_bruta).lower().strip()
    for chave in DICON_FALHAS.keys():
        if chave in classe_limpa:
            severidade = "".join(filter(str.isdigit, classe_limpa))
            return chave, severidade if severidade else "Única"
    return "desconhecida", "Desconhecida"

# ==============================================================================
# BARRA LATERAL (SELEÇÃO DE ALGORITMO + VELOCIDADE FIXA + SENSOR)
# ==============================================================================
st.sidebar.title("⚙️ Parâmetros Operacionais")

st.sidebar.markdown("### Modelo de Inteligência Artificial")
algoritmo_escolhido = st.sidebar.selectbox(
    "Algoritmo Classificador",
    options=["Random Forest", "XGBoost"],
    index=0
)
mapa_algoritmo = {"Random Forest": "rf", "XGBoost": "xgb"}
codigo_algoritmo = mapa_algoritmo[algoritmo_escolhido]

st.sidebar.markdown("### Configurações do Ativo")
motor_id = st.sidebar.text_input("Identificação do Motor/Bomba", value="Motor-2")

velocidade_escolhida = st.sidebar.selectbox(
    "Velocidade de Operação (%)",
    options=["50%", "75%", "100%"],
    index=1
)
velocidade_numerica = velocidade_escolhida.replace("%", "")

st.sidebar.markdown("### Posição de Medição")
posicao_sensor = st.sidebar.selectbox("Localização do Acelerômetro", [
    "1 - Carcaça do Motor (Lado NDE - Traseiro)",
    "2 - Mancal do Motor (Lado DE - Dianteiro)",
    "3 - Mancal do Motor (Direção Axial)",
    "4 - Carcaça do Mancal da Bomba (Lado Acoplamento)",
    "5 - Carcaça do Mancal da Bomba (Lado Impulsor)"
], index=3)

MAPA_SUFIXOS = {
    "1 - Carcaça do Motor (Lado NDE - Traseiro)": ["ch1", "canal1", "canal_1", "motor_traseiro"],
    "2 - Mancal do Motor (Lado DE - Dianteiro)": ["ch2", "canal2", "canal_2", "motor_dianteiro"],
    "3 - Mancal do Motor (Direção Axial)": ["ch3", "canal3", "canal_3", "axial"],
    "4 - Carcaça do Mancal da Bomba (Lado Acoplamento)": ["ch4", "canal4", "canal_4", "bomba_acoplamento"],
    "5 - Carcaça do Mancal da Bomba (Lado Impulsor)": ["ch5", "canal5", "canal_5", "bomba_impulsor"]
}
lista_sufixos_alvo = MAPA_SUFIXOS[posicao_sensor]

MAPA_ID_CANAIS = {
    "1 - Carcaça do Motor (Lado NDE - Traseiro)": 1,
    "2 - Mancal do Motor (Lado DE - Dianteiro)": 2,
    "3 - Mancal do Motor (Direção Axial)": 3,
    "4 - Carcaça do Mancal da Bomba (Lado Acoplamento)": 4,
    "5 - Carcaça do Mancal da Bomba (Lado Impulsor)": 5
}
id_canal_ia = MAPA_ID_CANAIS[posicao_sensor]

# Carrega dinamicamente o pacote correspondente à combinação escolhida
pacote_ia, nome_arquivo_modelo = carregar_pacote_ia(codigo_algoritmo, velocidade_numerica)

# ==============================================================================
# CORPO DA PÁGINA
# ==============================================================================
st.title("📊 Análise de Sinais Inteligente")
st.markdown("---")

if pacote_ia is None:
    st.error(f"Não foi possível carregar o arquivo `{nome_arquivo_modelo}` na pasta `modelo/`. Verifique se o arquivo existe.")
else:
    modelo = pacote_ia["modelo"]
    encoder = pacote_ia["encoder"]
    colunas_esperadas = pacote_ia["colunas"]

    st.info(
        f"🤖 Modelo ativo: **{algoritmo_escolhido}** (`{nome_arquivo_modelo}`) | "
        f"⚡ Velocidade: **{velocidade_escolhida}** | "
        f"📍 Sensor: **{posicao_sensor}** (`Canal_Sensor = {id_canal_ia}`)"
    )

    arquivo_upload = st.file_uploader("Faça o upload do arquivo CSV bruto de vibração", type=["csv"])

    if arquivo_upload is not None:
        if st.button("🚀 Iniciar Processamento Adaptativo"):
            with st.spinner("Aplicando filtro MAD, extraindo os 36 indicadores e consultando a IA..."):
                try:
                    arquivo_upload.seek(0)
                    df_header = pd.read_csv(arquivo_upload, nrows=0)
                    canal_final = None

                    for col_real in df_header.columns:
                        col_limpa = str(col_real).lower().strip()
                        if any(sufixo in col_limpa for sufixo in lista_sufixos_alvo):
                            canal_final = col_real
                            break

                    if canal_final is None:
                        if id_canal_ia < len(df_header.columns):
                            canal_final = df_header.columns[id_canal_ia]
                        else:
                            canal_final = df_header.columns[-1]

                    # Extrai os 36 indicadores já com o filtro MAD
                    X_pred = extrair_features_geral(arquivo_upload, canal_alvo=canal_final, id_canal_ia=id_canal_ia)

                    # Garante exatamente a ordem das 36 colunas salva no treinamento
                    X_pred = X_pred[colunas_esperadas]

                    # Predição numérica + decodificação pelo LabelEncoder + probabilidade
                    pred_num = modelo.predict(X_pred)
                    probabilidades = modelo.predict_proba(X_pred)[0]
                    confianca_pct = float(np.max(probabilidades) * 100)

                    predicao_str = str(encoder.inverse_transform(pred_num)[0]).lower().strip()
                    chave_falha, severidade_falha = tratar_classe_label(predicao_str)
                    info_falha = DICON_FALHAS.get(chave_falha, {"nome": "Desconhecida", "desc": "Padrão de falha não catalogado.", "cor": "orange"})

                    # Salva no estado da sessão para outras páginas do Streamlit (se houver)
                    arquivo_upload.seek(0)
                    st.session_state["sinal_bruto"] = pd.read_csv(arquivo_upload)[canal_final].values
                    st.session_state["features_atuais"] = X_pred
                    st.session_state["diagnostico_atual"] = info_falha["nome"]
                    st.session_state["severidade_atual"] = severidade_falha

                    # Painel de Diagnóstico
                    st.markdown("### Status do Diagnóstico do Ativo:")
                    if info_falha["cor"] == "green":
                        st.balloons()
                        st.success(f"## **{info_falha['nome']}** (Severidade: {severidade_falha}) — Confiança: {confianca_pct:.1f}%")
                    elif info_falha["cor"] == "orange":
                        st.warning(f"## **{info_falha['nome']}** (Severidade: {severidade_falha}) — Confiança: {confianca_pct:.1f}%")
                    else:
                        st.error(f"## **{info_falha['nome']}** (Severidade: {severidade_falha}) — Confiança: {confianca_pct:.1f}%")

                    st.write(f"**Ativo Monitorado:** *{motor_id}* operando a *{velocidade_escolhida}* de velocidade (Classificador: *{algoritmo_escolhido}*).")
                    st.write(f"**Análise da Localização:** Diagnóstico gerado com base no sensor posicionado em: *{posicao_sensor}*.")
                    st.write(f"**Laudo Mecânico Descritivo:** {info_falha['desc']}")

                    with st.expander("🔍 Detalhes Técnicos do Vetor de Indicadores"):
                        st.write("Vetor de 36 características extraído após filtragem robusta (MAD) e processado na memória RAM:")
                        st.dataframe(X_pred)

                except Exception as e:
                    st.error(f"Erro durante o processamento do arquivo: {e}")
