"""
modeling/reducer.py — Transformador para redução de features (PCA e LDA).
"""
from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class FeatureReducer(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'none', n_components: int = 5, logger: Any = None) -> None:
        self.method = method
        self.n_components = n_components
        self.logger = logger

    def _construir_redutor_interno(self):
        if self.method == 'none': return None
        if self.method == 'pca': 
            return PCA(n_components=self.n_components, random_state=42)
        if self.method == 'lda':
            # Em classificação binária, LDA suporta no máximo 1 componente
            return LDA(n_components=1)
        return None

    def fit(self, X, y=None) -> "FeatureReducer":
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.reducer_ = self._construir_redutor_interno()

        if self.reducer_ is None:
            self.feature_names_out_ = self.feature_names_in_
            return self

        # LDA exige o alvo (y) por ser supervisionado
        if self.method == 'lda':
            self.reducer_.fit(X, y)
        else:
            self.reducer_.fit(X)
            
        # Define nomes das colunas de saída
        if self.method == 'lda':
            self.feature_names_out_ = ['lda_0']
        else:
            self.feature_names_out_ = [f'pc_{i}' for i in range(self.reducer_.n_components_)]
            
        return self

    def transform(self, X, y=None):
        if self.reducer_ is None: return X
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_out = self.reducer_.transform(X_arr)
        
        return pd.DataFrame(
            X_out, 
            columns=self.feature_names_out_, 
            index=(X.index if isinstance(X, pd.DataFrame) else None)
        )

    @property
    def selected_features(self) -> list[str] | None:
        return getattr(self, 'feature_names_out_', None)