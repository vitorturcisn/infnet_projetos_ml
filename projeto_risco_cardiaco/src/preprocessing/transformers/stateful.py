"""
src/preprocessing/transformers/stateful.py — Transformadores que aprendem estatísticas.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
from src.preprocessing.base import BaseFeatureTransformer

class GroupMedianImputer(BaseFeatureTransformer):
    """Imputa valores ausentes usando a mediana do grupo."""
    def __init__(self, group_col: str, target_col: str, logger: Any = None) -> None:
        super().__init__(logger)
        self.group_col = group_col
        self.target_col = target_col
        self.medians_ = {}

    def fit(self, X: pd.DataFrame, y=None) -> GroupMedianImputer:
        # Só aprende se as colunas existirem (evita erro após SimpleImputer)
        if self.group_col in X.columns and self.target_col in X.columns:
            self.medians_ = X.groupby(self.group_col)[self.target_col].median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.medians_:
            return X
        X = X.copy()
        # Preenche os NaNs da coluna alvo usando o mapeamento do grupo
        X[self.target_col] = X[self.target_col].fillna(X[self.group_col].map(self.medians_))
        return X

class StandardScalerTransformer(BaseFeatureTransformer):
    """Aplica Z-score normalization: z = (x - mu) / sigma."""
    def __init__(self, columns: list[str], logger: Any = None) -> None:
        super().__init__(logger)
        self.columns = columns
        self.mean_ = {}
        self.std_ = {}

    def fit(self, X: pd.DataFrame, y=None) -> StandardScalerTransformer:
        # Se X for um array (de um imputer anterior sem set_output), 
        # tentamos converter para DF para não quebrar a lógica de nomes.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            # Se não temos nomes, usamos os índices numéricos das colunas como nomes
            # Mas o ideal é o model_factory passar 'set_output(transform="pandas")'
        
        # Filtra apenas as colunas que realmente existem no dado de entrada
        cols_presentes = [c for c in self.columns if c in X.columns]
        
        if cols_presentes:
            self.mean_ = X[cols_presentes].mean().to_dict()
            self.std_ = X[cols_presentes].std().to_dict()
        
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not self.mean_:
            return X
            
        X = X.copy()
        for col, mu in self.mean_.items():
            if col in X.columns:
                sigma = self.std_.get(col, 1.0)
                # O 1e-9 evita divisão por zero se o desvio padrão for nulo
                X[col] = (X[col] - mu) / (sigma + 1e-9)
        return X