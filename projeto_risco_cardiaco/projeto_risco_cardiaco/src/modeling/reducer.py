"""
modeling/reducer.py — Transformador para redução de features.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA

class FeatureReducer(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'none', n_components: int = 15, logger: Any = None) -> None:
        self.method = method
        self.n_components = n_components
        self.logger = logger

    def _construir_redutor_interno(self):
        if self.method == 'none': return None
        if self.method == 'pca': return PCA(n_components=self.n_components, random_state=42)
        return None

    def fit(self, X, y=None) -> "FeatureReducer":
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.reducer_ = self._construir_redutor_interno()

        if self.reducer_ is None:
            self.feature_names_out_ = self.feature_names_in_
            return self

        self.reducer_.fit(X)
        self.feature_names_out_ = [f'pc_{i}' for i in range(self.n_components)]
        return self

    def transform(self, X, y=None):
        if self.reducer_ is None: return X
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_out = self.reducer_.transform(X_arr)
        if self.feature_names_out_ is not None:
            return pd.DataFrame(X_out, columns=self.feature_names_out_, index=(X.index if isinstance(X, pd.DataFrame) else None))
        return X_out

    @property
    def selected_features(self) -> list[str] | None:
        return getattr(self, 'feature_names_out_', None)