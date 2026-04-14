from __future__ import annotations
from typing import Any
import pandas as pd
from src.preprocessing.base import BaseFeatureTransformer

class RatioFeatureTransformer(BaseFeatureTransformer):
    """Cria razões médicas (ex: Colesterol/IMC)."""

    def __init__(self, ratios: list[dict], logger: Any = None) -> None:
        super().__init__(logger)
        self.ratios = ratios

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for spec in self.ratios:
            name, num, den = spec["name"], spec["numerator"], spec["denominator"]
            X[name] = X[num] / X[den].replace(0, 0.001) # Evita divisão por zero
            self._log(f"RatioFeatureTransformer: Criada razão {name}")
        return X