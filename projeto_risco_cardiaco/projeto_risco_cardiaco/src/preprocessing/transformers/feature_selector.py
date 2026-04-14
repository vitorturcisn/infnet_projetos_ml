from __future__ import annotations
from typing import Any
import pandas as pd
from src.preprocessing.base import BaseFeatureTransformer

class FeatureSelector(BaseFeatureTransformer):
    """
    Seleciona apenas o subconjunto de colunas definido na configuração.
    """

    def __init__(self, features_to_keep: list[str], logger: Any = None) -> None:
        super().__init__(logger)
        self.features_to_keep = features_to_keep

    def fit(self, X: pd.DataFrame, y=None) -> FeatureSelector:
        """Valida se as colunas configuradas realmente existem no DataFrame."""
        ausentes = [c for c in self.features_to_keep if c not in X.columns]
        if ausentes:
            self._warn(f"FeatureSelector: {len(ausentes)} colunas da config não encontradas: {ausentes}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Filtra o DataFrame mantendo apenas as colunas desejadas."""
        disponiveis = [c for c in self.features_to_keep if c in X.columns]
        
        self._log(
            "FeatureSelector: Selecionadas %d features. Shape final: %s", 
            len(disponiveis), (len(X), len(disponiveis))
        )
        
        # Retorna uma cópia apenas com as colunas que decidimos manter
        return X[disponiveis].copy()