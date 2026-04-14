"""
modeling/model_factory.py — Fábrica de pipelines com Imputação Global e preservação de DataFrame.
"""
from __future__ import annotations

import importlib
from typing import Any

from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.impute import SimpleImputer
from src.preprocessing.transformers.stateful import StandardScalerTransformer
from src.modeling.reducer import FeatureReducer


def construir_modelo(model_cfg: dict, params_extras: dict | None = None) -> Any:
    """Instancia um modelo via importlib mesclando parâmetros padrão e extras."""
    modulo = importlib.import_module(model_cfg['module'])
    cls    = getattr(modulo, model_cfg['class'])
    params = dict(model_cfg.get('default_params') or {})
    if params_extras:
        params.update(params_extras)
    return cls(**params)


def construir_pipeline(
    model_cfg: dict, 
    params_modelo: dict | None, 
    params_reducer: dict | None, 
    pipe_cfg: dict,
) -> SklearnPipeline:
    """
    Constrói um sklearn Pipeline garantindo a passagem de DataFrames entre os steps.
    """
    steps = []
    
    # 1. SEGURANÇA TOTAL: Imputador global
    # O .set_output(transform="pandas") é a CHAVE para evitar o IndexError,
    # garantindo que o StandardScaler receba nomes de colunas.
    imputer = SimpleImputer(strategy='median').set_output(transform="pandas")
    steps.append(('global_imputer', imputer))
    
    # 2. ESCALONAMENTO
    # Agora o Scaler recebe um DataFrame e consegue ler self.columns sem erro.
    colunas_escala = pipe_cfg.get('scaling', {}).get('columns', [])
    if colunas_escala:
        steps.append(('scaler', StandardScalerTransformer(columns=colunas_escala)))

    # 3. REDUÇÃO E ESTIMADOR
    kw_reducer = params_reducer or {}
    steps.append(('reducer', FeatureReducer(**kw_reducer)))
    steps.append(('estimator', construir_modelo(model_cfg, params_modelo)))
    
    return SklearnPipeline(steps)