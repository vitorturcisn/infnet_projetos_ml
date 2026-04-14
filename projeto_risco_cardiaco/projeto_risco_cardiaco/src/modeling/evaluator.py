"""
modeling/evaluator.py — Avaliador do modelo no conjunto holdout.
"""
from __future__ import annotations

import logging
from typing import Any

from src.modeling.base import BaseEvaluator
from src.modeling.metrics import calcular_metricas


class HoldoutEvaluator(BaseEvaluator):
    """Avalia o modelo final no conjunto holdout (Classificação)."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger

    def avaliar(self, model: Any, X: Any, y: Any) -> dict:
        y_previsto = model.predict(X)
        y_previsto_proba = None
        if hasattr(model, "predict_proba"):
            y_previsto_proba = model.predict_proba(X)[:, 1]
            
        return calcular_metricas(y.values, y_previsto, y_previsto_proba)

    def diagnosticar_robustez(self, cv_auc: float, holdout_auc: float) -> str:
        """Compara o CV ROC-AUC com o Holdout ROC-AUC e emite diagnóstico."""
        # Na classificação (AUC), a métrica vai de 0.5 a 1.0. Uma queda de 0.05 é notável.
        delta = cv_auc - holdout_auc

        if self.logger:
            self.logger.info('── Análise de Robustez ──')
            self.logger.info('  CV ROC-AUC (média) : %.4f', cv_auc)
            self.logger.info('  Holdout ROC-AUC    : %.4f', holdout_auc)
            self.logger.info('  Δ (Queda)          : %.4f', delta)

        if delta < 0.03:
            diagnostico = 'BOA'
            if self.logger: self.logger.info('  Diagnóstico        : ✓ Generalização BOA (Δ < 0.03)')
        elif delta < 0.08:
            diagnostico = 'MODERADA'
            if self.logger: self.logger.info('  Diagnóstico        : ⚠ Generalização MODERADA')
        else:
            diagnostico = 'RUIM'
            if self.logger: self.logger.warning('  Diagnóstico        : ✗ Generalização RUIM (risco de overfitting!)')

        return diagnostico