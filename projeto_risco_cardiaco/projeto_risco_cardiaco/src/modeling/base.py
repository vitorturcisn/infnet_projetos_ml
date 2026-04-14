"""
modeling/base.py — Classes abstratas base do módulo de modelagem.

Define os contratos principais:
  - BaseOptimizer    : qualquer estratégia de otimização de hiperparâmetros
  - BaseEvaluator    : qualquer avaliador de modelo (holdout, CV externo, etc.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseOptimizer(ABC):
    """Contrato para otimizadores de hiperparâmetros."""

    @abstractmethod
    def otimizar(
        self,
        model_name: str,
        model_cfg: dict,
        X_tune: Any,
        y_tune: Any,
        pipe_cfg: dict,
        feat_red_cfg: dict,
    ) -> dict:
        """
        Executa a otimização e retorna um dicionário com os melhores parâmetros.
        """


class BaseEvaluator(ABC):
    """Contrato para avaliadores de modelo."""

    @abstractmethod
    def avaliar(self, model: Any, X: Any, y: Any) -> dict:
        """
        Avalia o modelo e retorna dicionário de métricas.

        Returns:
            dict com roc_auc, f1, precision, recall, accuracy
        """