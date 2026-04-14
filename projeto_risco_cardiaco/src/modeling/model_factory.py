"""
modeling/model_factory.py — Fábrica de pipelines atualizada para Reprodutibilidade.
"""
from __future__ import annotations
import importlib
from typing import Any
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.impute import SimpleImputer
from src.preprocessing.transformers.stateful import StandardScalerTransformer
from src.modeling.reducer import FeatureReducer

def construir_modelo(model_cfg: dict, params_extras: dict | None = None, seed: int = 42) -> Any:
    """Instancia o modelo garantindo a aplicação do random_seed global."""
    modulo = importlib.import_module(model_cfg['module'])
    cls    = getattr(modulo, model_cfg['class'])
    params = dict(model_cfg.get('default_params') or {})
    
    # Injeta o random_state se o modelo aceitar (quase todos do sklearn/xgboost/lgbm aceitam)
    if 'random_state' not in params:
        params['random_state'] = seed
        
    if params_extras:
        params.update(params_extras)
    return cls(**params)

def construir_pipeline(
    model_cfg: dict, 
    params_modelo: dict | None, 
    params_reducer: dict | None, 
    pipe_cfg: dict,
    seed: int = 42 # <── Adicionado para consistência
) -> SklearnPipeline:
    steps = []
    
    # 1. Imputador global (Garante DataFrames para os próximos steps)
    imputer = SimpleImputer(strategy='median').set_output(transform="pandas")
    steps.append(('global_imputer', imputer))
    
    # 2. Escalonamento
    colunas_escala = pipe_cfg.get('scaling', {}).get('columns', [])
    if colunas_escala:
        steps.append(('scaler', StandardScalerTransformer(columns=colunas_escala)))

    # 3. Redução de Dimensionalidade (PCA/LDA integrados aqui)
    kw_reducer = params_reducer or {}
    steps.append(('reducer', FeatureReducer(**kw_reducer)))
    
    # 4. Estimador Final
    steps.append(('estimator', construir_modelo(model_cfg, params_modelo, seed=seed)))
    
    return SklearnPipeline(steps)