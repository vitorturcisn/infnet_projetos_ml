"""
utils/model_utils.py — Cliente MLflow local para Risco Cardíaco.
"""
from __future__ import annotations
import mlflow
import mlflow.pyfunc
import pandas as pd

_NOME_MODELO: str = "cardiac-risk-best"
_N_FOLDS_CV: int  = 5  # No nosso modeling.yaml usamos 5 folds

def carregar_modelo(db_uri: str) -> mlflow.pyfunc.PyFuncModel:
    mlflow.set_tracking_uri(db_uri)
    # Carrega a versão 'latest' (mais recente) registrada
    return mlflow.pyfunc.load_model(f"models:/{_NOME_MODELO}/latest")

def prever_individual(features_df: pd.DataFrame, modelo: mlflow.pyfunc.PyFuncModel) -> dict:
    # Retorna tanto a classe (0/1) quanto a probabilidade
    classe = modelo.predict(features_df)[0]
    probabilidade = 0.0
    if hasattr(modelo, "predict_proba"):
        probabilidade = modelo.predict_proba(features_df)[0][1]
    return {"classe": int(classe), "prob": float(probabilidade)}

def obter_params_performance(db_uri: str) -> dict:
    cliente = mlflow.MlflowClient(tracking_uri=db_uri)
    versoes = cliente.search_model_versions(f"name='{_NOME_MODELO}'")
    if not versoes:
        raise ValueError("Nenhuma versão de modelo encontrada.")
    
    ultima = max(versoes, key=lambda v: int(v.version))
    run = cliente.get_run(ultima.run_id)
    m = run.data.metrics
    
    return {
        "auc": m.get("cv_roc_auc_mean", 0.5),
        "recall": m.get("holdout_recall", 0.0),
        "versao": ultima.version,
        "run_id": ultima.run_id
    }