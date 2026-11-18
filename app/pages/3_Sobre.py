import streamlit as st

st.set_page_config(page_title="Sobre o Projeto", page_icon="ℹ️", layout="wide")

st.title("ℹ️ UFSJ - TCC II: Sistema Inteligente para Diagnóstico de Falhas em Motobombas Centrífugas")
st.markdown("---")

st.info("**Trabalho de Conclusão de Curso (TCC II) — Engenharia Mecânica**")
st.write("**Autor:** Rodrigo José de Oliveira Santos Júnior")
st.write(
    "**Tema:** Desenvolvimento de Modelo Comparativo de Aprendizado de Máquina Supervisionado "
    "Aplicado ao Diagnóstico Descritivo e de Severidade em Sistemas Hidráulicos de Fluxo Centrífugo"
)

st.markdown("---")
st.subheader("📋 Ficha Técnica da Metodologia e Arquitetura")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**1. Conjunto Experimental e Aquisição**")
    st.write(
        "- **Equipamento:** Conjunto motobomba centrífuga monitorado por 5 acelerômetros piezoelétricos.\n"
        "- **Taxa de Amostragem:** 20 kHz ($f_s = 20.000\\text{ Hz}$).\n"
        "- **Regimes Avaliados:** 50%, 75% e 100% da Velocidade operacional.\n"
        "- **Escopo Diagnóstico:** Classificação multiclasse descritiva e de severidade (até 23 condições operacionais)."
    )

with col2:
    st.markdown("**2. Processamento e Inteligência Artificial**")
    st.write(
        "- **Filtragem Robusta:** Remoção de *outliers* via Desvio Absoluto Mediano (MAD) e *Z-Score* modificado.\n"
        "- **Engenharia de Atributos:** Extração de 36 indicadores físicos nos domínios do tempo, frequência e outros (FFT entre 5 Hz e 2.000 Hz).\n"
        "- **Classificadores Otimizados:** *Random Forest* vs. *XGBoost* ajustados via *GridSearchCV* com validação cruzada estratificada ($k = 10$).\n"
        "- **Tecnologias:** Python, Scikit-Learn, XGBoost, SciPy, Pandas, Plotly e Streamlit."
    )
