import logging
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BaseFeatureTransformer(BaseEstimator, TransformerMixin, ABC):
    """Classe base para todos os transformadores do pipeline."""
    
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        """Emite um log informativo."""
        if self.logger:
            self.logger.info(msg, *args)

    def _warn(self, msg: str, *args: Any) -> None:
        """Emite um aviso (warning) se algo não estiver certo."""
        if self.logger:
            self.logger.warning(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "BaseFeatureTransformer":
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        pass