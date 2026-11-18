import streamlit as st
import os

# Configuração global da aplicação
st.set_page_config(page_title="TCC Diagnóstico de Bombas", page_icon="⚙️", layout="wide")

# Caminhos dinâmicos para verificar os 6 modelos treinados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.path.abspath(os.path.join(BASE_DIR, "..", "modelo"))

modelos_esperados = [
    "modelo_rf_50pct.pkl", "modelo_rf_75pct.pkl", "modelo_rf_100pct.pkl",
    "modelo_xgb_50pct.pkl", "modelo_xgb_75pct.pkl", "modelo_xgb_100pct.pkl"
]

modelos_encontrados = [m for m in modelos_esperados if os.path.exists(os.path.join(PASTA_MODELOS, m))]
modelos_faltando = [m for m in modelos_esperados if m not in modelos_encontrados]

# ==============================================================================
# CABEÇALHO E APRESENTAÇÃO DO SISTEMA
# ==============================================================================
st.title("⚙️ Sistema Inteligente para Diagnóstico de Falhas em Motobombas Centrífugas")
st.subheader("Plataforma de Manutenção Preditiva Baseada em Análise de Vibrações e Aprendizado de Máquina")
st.markdown("---")

st.write("Bem-vindo ao sistema de diagnóstico automatizado desenvolvido para o **Trabalho de Conclusão de Curso (TCC) em Engenharia Mecânica**.")
st.write(
    "Esta aplicação realiza a filtragem estatística robusta via **Desvio Absoluto Mediano (MAD)**, "
    "extrai **29 indicadores físicos nos domínios do tempo e da frequência (FFT) e mais 7 outros** e utiliza modelos "
    "de classificação supervisionada (**Random Forest** e **XGBoost**) otimizados para diagnosticar "
    "falhas mecânicas de um sistema motobomba sob diferentes velocidades de operação."
)

# ==============================================================================
# VERIFICAÇÃO DOS MODELOS NO SERVIDOR
# ==============================================================================
st.markdown("### 🧠 Status dos Modelos de Inteligência Artificial")

if len(modelos_encontrados) == 6:
    st.success(
        "✅ **Todos os 6 modelos treinados (Random Forest e XGBoost para 50%, 75% e 100% de velocidade) "
        "foram detectados na pasta `modelo` e estão prontos para operação.**"
    )
elif len(modelos_encontrados) > 0:
    st.warning(
        f"⚠️ **{len(modelos_encontrados)} de 6 modelos encontrados.** "
        f"Arquivos ausentes: `{', '.join(modelos_faltando)}`"
    )
else:
    st.error(
        "❌ Nenhum modelo `.pkl` foi encontrado na pasta `modelo`. "
        "Por favor, execute o script de treinamento antes de iniciar as análises."
    )

# ==============================================================================
# CARDS RESUMO DA ARQUITETURA DO PROJETO
# ==============================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📡 Processamento de Sinais")
    st.write(
        "- Frequência de amostragem: **20 kHz**\n"
        "- Filtro de *outliers* via **MAD**\n"
        "- Banda espectral: **5 Hz a 2.000 Hz**"
    )

with col2:
    st.markdown("#### 📐 Engenharia de Atributos")
    st.write(
        "- **16** indicadores no domínio do tempo\n"
        "- **13** indicadores espectrais (FFT)\n"
        "- **7** indicadores adicionais"
    )

with col3:
    st.markdown("#### 🤖 Classificação Multiclasse")
    st.write(
        "- **Random Forest** vs. **XGBoost**\n"
        "- Velocidades: **50%**, **75%** e **100%**\n"
        "- Diagnóstico de até **23 condições**"
    )

st.markdown("---")
st.info("💡 Utilize o menu lateral esquerdo para navegar entre as páginas de **Analise** (Diagnóstico de Sinais) e **Resultados** (Gráficos Dinâmicos e Auditoria dos Modelos).")
