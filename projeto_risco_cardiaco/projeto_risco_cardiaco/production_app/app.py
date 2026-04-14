import sys
from pathlib import Path
import streamlit as st

# ── O SEGREDO DO SUCESSO ──
# Adicionamos a raiz do projeto ao sys.path antes de qualquer outra coisa
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = str(_HERE.parent)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Agora sim configuramos a página
st.set_page_config(page_title="App MLOps - Risco Cardíaco", page_icon="🏥", layout="wide")

st.title("🏥 Sistema MLOps de Diagnóstico Cardíaco")

st.markdown("""
### Bem-vindo ao Sistema
Utilize o menu lateral para navegar entre a **Predição** e o **Monitoramento**.
""")

if st.button("Ir para Predição"):
    st.switch_page("pages/1_Predicao.py")