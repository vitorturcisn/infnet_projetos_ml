"""
modeling/metrics.py — Funções de cálculo e agregação de métricas de classificação.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

def calcular_metricas(
    y_verdadeiro: np.ndarray, 
    y_previsto: np.ndarray, 
    y_previsto_proba: np.ndarray | None = None
) -> dict:
    """Calcula as métricas de classificação binária usadas no pipeline."""
    
    # Métricas baseadas na classe cravada (0 ou 1)
    acc = float(accuracy_score(y_verdadeiro, y_previsto))
    f1 = float(f1_score(y_verdadeiro, y_previsto, zero_division=0))
    precision = float(precision_score(y_verdadeiro, y_previsto, zero_division=0))
    recall = float(recall_score(y_verdadeiro, y_previsto, zero_division=0))
    
    # ROC-AUC precisa da probabilidade da classe 1 (ex: 0.85 chance de estar doente)
    roc_auc = 0.5 # Baseline aleatório se falhar
    if y_previsto_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_verdadeiro, y_previsto_proba))
        except ValueError:
            pass # Se o modelo falhar em prever proba, fica com o baseline
            
    return {
        'roc_auc': roc_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': acc
    }

def agregar_metricas_folds(fold_metrics: list[dict]) -> dict:
    """Agrega métricas de todos os folds em média e desvio padrão."""
    df = pd.DataFrame(fold_metrics)
    resultado = {}
    for col in ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']:
        if col in df.columns:
            resultado[f'cv_{col}_mean'] = float(df[col].mean())
            resultado[f'cv_{col}_std']  = float(df[col].std())
    return resultado