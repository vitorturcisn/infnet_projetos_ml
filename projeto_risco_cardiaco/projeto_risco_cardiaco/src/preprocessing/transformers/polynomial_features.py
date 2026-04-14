from __future__ import annotations
from typing import Any
import pandas as pd
from src.preprocessing.base import BaseFeatureTransformer

class PolynomialFeatureTransformer(BaseFeatureTransformer):
    """Cria termos de interação (ex: Idade x IMC)."""

    def __init__(self, poly_config: list[dict], logger: Any = None) -> None:
        super().__init__(logger)
        self.poly_config = poly_config

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for spec in self.poly_config:
            name, cols = spec["name"], spec["columns"]
            if len(cols) == 2:
                X[name] = X[cols[0]] * X[cols[1]]
                self._log(f"PolynomialFeatureTransformer: Interação {name} criada.")
        return X