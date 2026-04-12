"""
pages/2_Monitoramento.py — Dashboard: monitoramento de classificação de Risco Cardíaco.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ── Bootstrap de paths ────────────────────────────────────────────────────────
_PAGE_DIR     = Path(__file__).resolve().parent
_APP_DIR      = _PAGE_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import obter_parquet_features
from utils.model_utils import carregar_modelo

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Monitoramento de Risco", page_icon="📡", layout="wide")

st.title("📡 Dashboard de Monitoramento do Modelo")
st.markdown(
    """
    Simula o monitoramento de produção em lote. O sistema divide os dados históricos em "janelas temporais" (lotes) 
    para verificar se a assertividade médica do modelo (AUC e Recall) permanece estável.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
_URI_PADRAO = f"sqlite:///{_PROJECT_ROOT / 'mlruns.db'}"

with st.sidebar:
    st.header("⚙️ Configurações")
    db_uri = st.text_input("URI do banco MLflow", value=_URI_PADRAO)
    n_amostras = st.slider("Amostras para Teste", 100, 1000, 500, 50)
    n_lotes = st.slider("Número de Lotes (Simulação)", 5, 20, 10)
    janela_movel = st.slider("Janela de Média Móvel", 2, 5, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Funções de Cálculo (Adaptadas para Classificação)
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_metricas_classificacao(y_true, y_pred, y_prob):
    """Calcula métricas vitais para o contexto médico."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
        
    return {
        "auc": auc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred)
    }

def _plotar_serie_temporal(ax, lotes, valores, movel, nome, cor):
    ax.plot(lotes, valores, "o-", color=cor, alpha=0.3, label="Lote Individual")
    ax.plot(lotes, movel, "-", color=cor, linewidth=3, label=f"Tendência (Média Móvel)")
    ax.set_title(nome.upper(), fontsize=12, fontweight="bold", color="white")
    ax.grid(True, alpha=0.2)
    ax.legend()

# ─────────────────────────────────────────────────────────────────────────────
# Execução
# ─────────────────────────────────────────────────────────────────────────────
if st.button("▶️ Iniciar Análise de Drift", type="primary", use_container_width=True):

    with st.spinner("Analisando performance dos lotes..."):
        # 1. Carrega dados e modelo
        df = obter_parquet_features()
        modelo = carregar_modelo(db_uri)
        
        # 2. Amostragem
        df_amostra = df.sample(n_amostras, random_state=42).reset_index(drop=True)
        y_true = df_amostra['target_numeric'].values
        X_amostra = df_amostra.drop(columns=['target_numeric', 'Heart Disease Status'], errors='ignore')

        # 3. Predição
        y_pred = modelo.predict(X_amostra)
        y_prob = modelo.predict_proba(X_amostra)[:, 1] if hasattr(modelo, "predict_proba") else y_pred

        # 4. Cálculo por Lote
        tamanho_lote = len(X_amostra) // n_lotes
        metricas_lista = []
        
        for i in range(n_lotes):
            start, end = i*tamanho_lote, (i+1)*tamanho_lote
            m = _calcular_metricas_classificacao(y_true[start:end], y_pred[start:end], y_prob[start:end])
            m["lote"] = i + 1
            metricas_lista.append(m)

        df_met = pd.DataFrame(metricas_lista).set_index("lote")
        df_movel = df_met.rolling(window=janela_movel, min_periods=1).mean()

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 1 — KPIs Gerais
    # ═══════════════════════════════════════════════════════════════════════════
    m_geral = _calcular_metricas_classificacao(y_true, y_pred, y_prob)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC Médio", f"{m_geral['auc']:.4f}")
    col2.metric("Recall (Sensibilidade)", f"{m_geral['recall']:.2%}", help="Capacidade de detectar doentes reais")
    col3.metric("F1-Score", f"{m_geral['f1']:.4f}")
    col4.metric("Acurácia Geral", f"{m_geral['accuracy']:.2%}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 2 — Gráficos de Tendência
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📈 Estabilidade das Métricas Médicas")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.patch.set_facecolor("#0e1117")
    
    config_plots = [
        ("auc", axes[0,0], "#3498db", "Área sob Curva (AUC)"),
        ("recall", axes[0,1], "#e74c3c", "Recall (Detecção de Doentes)"),
        ("f1", axes[1,0], "#2ecc71", "F1-Score (Equilíbrio)"),
        ("accuracy", axes[1,1], "#f1c40f", "Acurácia")
    ]

    for m_key, ax, cor, titulo in config_plots:
        ax.set_facecolor("#1a1a2e")
        _plotar_serie_temporal(ax, df_met.index, df_met[m_key], df_movel[m_key], titulo, cor)
        ax.tick_params(colors="white")

    plt.tight_layout()
    st.pyplot(fig)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 3 — Diagnóstico de Erro
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 Matriz de Confusão Acumulada")
        fig_cm, ax_cm = plt.subplots()
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Saudável", "Doente"]).plot(ax=ax_cm, cmap="Reds")
        st.pyplot(fig_cm)

    with col_right:
        st.subheader("📋 Tabela de Auditoria por Lote")
        st.dataframe(df_met.style.background_gradient(cmap="RdYlGn", subset=["auc", "recall"]), use_container_width=True)