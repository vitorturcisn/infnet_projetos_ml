from __future__ import annotations
from typing import Any
import pandas as pd
from src.preprocessing.base import BaseFeatureTransformer

class BinaryFlagTransformer(BaseFeatureTransformer):
    """Cria colunas binárias marcando zonas de risco (ex: Hipertensão, Obesidade)."""

    def __init__(self, flags: list[dict], logger: Any = None) -> None:
        super().__init__(logger)
        self.flags = flags

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for spec in self.flags:
            col, val, new_col = spec["column"], spec["value"], spec["new_column"]
            if col not in X.columns:
                self._warn(f"BinaryFlagTransformer: '{col}' ausente — flag '{new_col}' ignorada.")
                continue
            
            X[new_col] = (X[col] > val).astype(int)
            self._log(f"BinaryFlagTransformer: {new_col} criado (threshold > {val})")
        return X