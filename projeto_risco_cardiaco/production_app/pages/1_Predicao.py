import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Bootstrap de paths
_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
if str(_APP_DIR) not in sys.path:
    sys.path.append(str(_APP_DIR))

from utils.pipeline_utils import preprocessar_entradas
from utils.model_utils import carregar_modelo, prever_individual, obter_params_performance

st.set_page_config(page_title="Predição de Risco Cardíaco", page_icon="❤️", layout="wide")

st.title("❤️ Diagnóstico de Risco Cardíaco")
st.markdown("Insira os dados do paciente para calcular a probabilidade de doença cardíaca.")

with st.sidebar:
    db_uri = st.text_input("MLflow SQLite URI", value=f"sqlite:///{_PROJECT_ROOT}/mlruns.db")
    # MODO DETETIVE: Marque isso para ver as colunas que o modelo exige
    debug_mode = st.checkbox("Exibir 'DNA' do modelo (Debug)")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Idade", 18, 100, 45)
    gender = st.selectbox("Gênero", ["Male", "Female"])
    bmi = st.number_input("IMC (BMI)", 10.0, 50.0, 24.0)
    stress = st.selectbox("Nível de Estresse", ["Low", "Medium", "High"])
    # ADICIONADO: Campos que o modelo exige
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

with col2:
    chol = st.number_input("Colesterol total", 100, 500, 200)
    trig = st.number_input("Triglicerídeos", 50, 800, 150)
    crp = st.number_input("Nível de CRP", 0.0, 10.0, 1.0)
    homo = st.number_input("Homocisteína", 0.0, 50.0, 10.0)
    # ADICIONADO: Campo que o modelo exige
    smoking = st.selectbox("Fumante", ["No", "Yes"])

if st.button("🔮 Calcular Risco", type="primary", use_container_width=True):
    raw_data = {
        "Age": age, "Gender": gender, "BMI": bmi, "Stress Level": stress,
        "Cholesterol Level": chol, "Triglyceride Level": trig, 
        "CRP Level": crp, "Homocysteine Level": homo,
        "Diabetes": diabetes, "Smoking": smoking # Agora os dados chegam ao pipeline!
    }
    
    try:
        # 1. Carregar o Modelo
        modelo = carregar_modelo(db_uri)
        
        # 2. MODO DETETIVE (Versão Ninja para SklearnModelWrapper)
        if debug_mode:
            st.warning("### 🔍 Tentando extrair a ordem das colunas...")
            try:
                # Caminho 1: Assinatura do MLflow (O mais oficial)
                nomes = [f.name for f in modelo.metadata.get_input_schema().inputs]
                st.code(nomes)
                st.info("Sucesso via MLflow Schema!")
            except:
                try:
                    # Caminho 2: Por baixo do wrapper do SklearnModelWrapper
                    # Geraldamente o modelo real está em .sklearn_model ou .model
                    inner_model = modelo._model_impl.sklearn_model
                    st.code(list(inner_model.feature_names_in_))
                    st.info("Sucesso via Sklearn Model interno!")
                except:
                    try:
                        # Caminho 3: Se for XGBoost/LGBM escondido
                        st.code(list(modelo._model_impl.model.feature_names_in_))
                    except Exception as e:
                        st.error(f"Ainda não consegui. Erro: {e}")
                        st.write("Dica: Olhe o arquivo 'MLmodel' na pasta da run no MLflow UI.")
        
        # 3. Processar Entradas
        features = preprocessar_entradas(raw_data)
        
        # 4. Predição
        res = prever_individual(features, modelo)
        perf = obter_params_performance(db_uri)

        st.divider()
        prob = res['prob'] * 100
        
        if prob > 50:
            st.error(f"### ALTO RISCO DETECTADO: {prob:.1f}%")
        else:
            st.success(f"### BAIXO RISCO: {prob:.1f}%")
            
        st.info(f"Modelo: {perf['versao']} | AUC Treino: {perf['auc']:.2f}")

    except Exception as e:
        st.error(f"⚠️ Erro de Compatibilidade: {e}")
        st.info("Dica: Ative o 'Modo Detetive' na lateral para ver se a ordem das colunas no seu 'pipeline_utils.py' está igual à do modelo.")