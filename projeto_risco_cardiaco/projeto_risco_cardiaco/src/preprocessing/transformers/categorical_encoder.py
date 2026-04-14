from __future__ import annotations
from typing import Any
import pandas as pd
from src.preprocessing.base import BaseFeatureTransformer

class CardiacCategoricalEncoder(BaseFeatureTransformer):
    """
    Encoder Robusto para MLOps.
    Garante que os códigos numéricos sejam consistentes entre treino e produção.
    """
    def __init__(self, enc_config: dict, logger: Any = None) -> None:
        super().__init__(logger)
        self.enc_config = enc_config
        # Mapeamentos fixos: A "Lei" do seu modelo
        self.mappings = {
            "Stress Level": {"Low": 0, "Medium": 1, "High": 2},
            "Smoking": {"No": 0, "Yes": 1},
            "Gender": {"Female": 0, "Male": 1}
        }

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        columns = self.enc_config.get("columns", [])
        prefix = self.enc_config.get("one_hot_prefix", "cat")
        X = X.copy()

        for col in columns:
            if col not in X.columns:
                continue

            # 1. Encoding Ordinal DETERMINÍSTICO
            if col in self.mappings:
                X[f"{col}_encoded"] = X[col].map(self.mappings[col]).fillna(-1).astype(int)
            else:
                X[f"{col}_encoded"] = pd.factorize(X[col])[0]

            # 2. One-hot Encoding (opcional, mas mantido para compatibilidade)
            full_prefix = f"{prefix}_{col}"
            dummies = pd.get_dummies(X[col], prefix=full_prefix)
            X = pd.concat([X, dummies], axis=1)

        return X