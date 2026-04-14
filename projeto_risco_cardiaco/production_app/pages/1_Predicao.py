import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Bootstrap de paths (Mantido)
_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
if str(_APP_DIR) not in sys.path:
    sys.path.append(str(_APP_DIR))

from utils.pipeline_utils import preprocessar_entradas
from utils.model_utils import carregar_modelo, prever_individual, obter_params_performance

st.set_page_config(page_title="Predição de Risco Cardíaco", page_icon="❤️", layout="wide")

# --- LÓGICA DE APOIO À DECISÃO CLÍNICA ---
# Definimos um limiar conservador (0.30) para priorizar o Recall, 
# evitando que pacientes de risco sejam ignorados[cite: 358, 562].
LIMIAR_DESCONFIADO = 0.30 

def exibir_fatores_risco(dados_originais):
    """Exibe alertas baseados na relevância clínica detectada no EDA e modelos anteriores."""
    st.markdown("#### 📋 Fatores de Risco Identificados:")
    encontrou_fator = False

    # BMI: Variável raiz da Árvore de Decisão e Top 1 no Random Forest [cite: 472, 544]
    if dados_originais["BMI"] > 25:
        st.write(f"- **IMC Elevado ({dados_originais['BMI']:.1f}):** Identificado como o critério de corte mais significativo para separação de grupos de risco[cite: 473, 545].")
        encontrou_fator = True

    # Smoking: Maior peso isolado no modelo Perceptron Otimizado 
    if dados_originais["Smoking"] == "Yes":
        st.write("- **Tabagismo Ativo:** Fator de risco com o maior peso absoluto na inclinação do hiperplano de decisão linear[cite: 369, 547].")
        encontrou_fator = True

    # CRP Level: Principal preditor não-linear no modelo Ensemble 
    if dados_originais["CRP Level"] > 3.0:
        st.write(f"- **PCR Elevada ({dados_originais['CRP Level']}):** Marcador inflamatório crucial para diagnósticos não-lineares complexos.")
        encontrou_fator = True

    # Stress Level: Forte sinalizador de risco silencioso detectado no EDA atual
    if dados_originais["Stress Level"] in ["Medium", "High"]:
        st.write(f"- **Estresse ({dados_originais['Stress Level']}):** Níveis elevados impactam a estabilidade dos indicadores metabólicos.")
        encontrou_fator = True

    if not encontrou_fator:
        st.write("- Nenhum fator isolado crítico detectado; o risco provém da interação multivariada dos indicadores.")

# --- INTERFACE (Mantida com ajustes de campos) ---
st.title("❤️ Diagnóstico de Risco Cardíaco")
st.markdown("Insira os dados do paciente para calcular a probabilidade de doença cardíaca.")

with st.sidebar:
    db_uri = st.text_input("MLflow SQLite URI", value=f"sqlite:///{_PROJECT_ROOT}/mlruns.db")
    debug_mode = st.checkbox("Exibir 'DNA' do modelo (Debug)")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Idade", 18, 100, 45)
    gender = st.selectbox("Gênero", ["Male", "Female"])
    bmi = st.number_input("IMC (BMI)", 10.0, 50.0, 24.0)
    stress = st.selectbox("Nível de Estresse", ["Low", "Medium", "High"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

with col2:
    chol = st.number_input("Colesterol total", 100, 500, 200)
    trig = st.number_input("Triglicerídeos", 50, 800, 150)
    crp = st.number_input("Nível de CRP", 0.0, 10.0, 1.0)
    homo = st.number_input("Homocisteína", 0.0, 50.0, 10.0)
    smoking = st.selectbox("Fumante", ["No", "Yes"])

if st.button("🔮 Calcular Risco", type="primary", use_container_width=True):
    raw_data = {
        "Age": age, "Gender": gender, "BMI": bmi, "Stress Level": stress,
        "Cholesterol Level": chol, "Triglyceride Level": trig, 
        "CRP Level": crp, "Homocysteine Level": homo,
        "Diabetes": diabetes, "Smoking": smoking 
    }
    
    try:
        modelo = carregar_modelo(db_uri, version=1)
        
        # DEBUG (Mantido)
        if debug_mode:
            st.warning("### 🔍 DNA do Modelo")
            try:
                inner_model = modelo._model_impl.sklearn_model
                st.code(list(inner_model.feature_names_in_))
            except:
                st.write("Não foi possível extrair nomes das colunas.")

        # 3. Processar Entradas e Predição
        features = preprocessar_entradas(raw_data)
        res = prever_individual(features, modelo)
        perf = obter_params_performance(db_uri)

        st.divider()
        prob = res['prob'] 
        
        # --- LÓGICA DE TRIAGEM AGRESSIVA ---
        if prob >= LIMIAR_DESCONFIADO:
            st.error(f"### ⚠️ CHANCES DE NECESSIDADE DE EXAMES: {prob*100:.1f}%")
            st.markdown("""
            **Orientação:** Este modelo atua como triagem inicial. Embora a probabilidade não seja absoluta, 
            a presença de indicadores específicos justifica uma investigação clínica aprofundada para evitar 
            Falsos Negativos[cite: 559, 562].
            """)
            exibir_fatores_risco(raw_data)
        else:
            st.success(f"### ✅ BAIXO RISCO: {prob*100:.1f}%")
            st.write("Os indicadores atuais sugerem estabilidade clínica.")
            
        st.info(f"Modelo: {perf['versao']} | AUC Treino: {perf['auc']:.2f} | Limiar de Triagem: {LIMIAR_DESCONFIADO}")

    except Exception as e:
        st.error(f"⚠️ Erro: {e}")