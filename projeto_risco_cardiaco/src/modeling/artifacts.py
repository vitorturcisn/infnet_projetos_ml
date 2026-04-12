"""
modeling/artifacts.py — Gerador de artefatos diagnósticos de Classificação Binária.

Plots gerados (configuráveis em modeling.yaml -> artifacts.plots):
  confusion_matrix         — Matriz de Verdadeiros/Falsos Positivos e Negativos
  roc_curve                — Trade-off entre True Positive Rate e False Positive Rate
  precision_recall_curve   — Trade-off entre Precisão e Recall (ótimo para desbalanceados)
  feature_importance       — Top-20 features por importância
  optuna_history           — Histórico de otimização
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')  # backend não-interativo: salva em arquivo sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


class ArtifactGenerator:
    """Gera e salva plots diagnósticos do melhor modelo de classificação."""

    def __init__(
        self,
        output_dir: Path,
        artifacts_cfg: dict,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_dir   = Path(output_dir)
        self.artifacts_cfg = artifacts_cfg
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _salvar(self, fig: Any, nome: str) -> Path | None:
        """Salva a figura em disco e fecha."""
        caminho = self.output_dir / nome
        try:
            fig.savefig(caminho, dpi=120, bbox_inches='tight')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Falha ao salvar plot %s: %s', nome, exc)
            return None
        finally:
            plt.close(fig)
        if self.logger:
            self.logger.info('Plot salvo: %s', caminho)
        return caminho

    def _extrair_importancia_features(
        self, model: Any, feature_names: list[str], X_val: pd.DataFrame, y_val: pd.Series,
    ) -> pd.Series:
        """Extrai importância de features adaptado para Classificação."""
        if isinstance(model, SklearnPipeline):
            estimador = model.named_steps['estimator']
            nomes_imp = feature_names
        else:
            estimador = model
            nomes_imp = feature_names

        if hasattr(estimador, 'feature_importances_'):
            return pd.Series(estimador.feature_importances_, index=nomes_imp)
        elif hasattr(estimador, 'coef_'):
            coef = np.abs(estimador.coef_[0]) # Regressão Logística retorna array 2D
            return pd.Series(coef, index=nomes_imp)
        else:
            # Fallback usando AUC como métrica de permutação
            amostra = min(2000, len(X_val))
            idx = np.random.default_rng(42).choice(len(X_val), amostra, replace=False)
            r = permutation_importance(
                model, X_val.iloc[idx], y_val.iloc[idx],
                n_repeats=5, random_state=42, n_jobs=-1, scoring='roc_auc'
            )
            return pd.Series(r.importances_mean, index=feature_names)

    # ── Plots de Classificação ────────────────────────────────────────────────

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Path | None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax, cmap='Blues', colorbar=False, 
            display_labels=["Saudável", "Doente"]
        )
        ax.set_title(f'Matriz de Confusão — {model_name}')
        plt.tight_layout()
        return self._salvar(fig, 'confusion_matrix.png')

    def plot_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray, model_name: str) -> Path | None:
        fig, ax = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_true, y_pred_proba, ax=ax, color='darkorange', name=model_name)
        ax.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.50)')
        ax.set_xlabel('False Positive Rate (Taxa de Falsos Alarmes)')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.set_title(f'Curva ROC — {model_name}')
        ax.legend(loc="lower right")
        plt.tight_layout()
        return self._salvar(fig, 'roc_curve.png')

    def plot_precision_recall_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray, model_name: str) -> Path | None:
        fig, ax = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba, ax=ax, color='teal', name=model_name)
        ax.set_xlabel('Recall (Doentes encontrados)')
        ax.set_ylabel('Precision (Acertos quando diz doente)')
        ax.set_title(f'Curva Precision-Recall — {model_name}')
        ax.legend(loc="lower left")
        plt.tight_layout()
        return self._salvar(fig, 'precision_recall_curve.png')

    def plot_feature_importance(self, model: Any, feature_names: list[str], X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Path | None:
        try:
            importancia = self._extrair_importancia_features(model, feature_names, X_train, y_train)
            top20 = importancia.nlargest(20).sort_values()
            fig, ax = plt.subplots(figsize=(10, 8))
            top20.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
            ax.set_xlabel('Importância / Peso')
            ax.set_title(f'Top Features (Preditores de Risco) — {model_name}')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            return self._salvar(fig, 'feature_importance.png')
        except Exception as exc:
            if self.logger: self.logger.warning('Feature importance falhou: %s', exc)
            return None

    def plot_optuna_history(self, study: Any, model_name: str) -> Path | None:
        if len(study.trials) <= 1: return None
        try:
            import optuna
            fig_ax = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig_ax.figure.set_size_inches(10, 5)
            return self._salvar(fig_ax.figure, f'optuna_history_{model_name}.png')
        except Exception as exc:
            return None

    # ── API de alto nível ─────────────────────────────────────────────────────

    def gerar_diagnosticos_modelo(
        self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, fold_metrics: list[dict],
    ) -> dict[str, Path | None]:
        
        y_pred = model.predict(X_train)
        
        # Tenta pegar as probabilidades (necessário para ROC). Se o modelo não suportar, falha suavemente.
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_train)[:, 1]

        # Evitando dependência circular aqui, a importação virá do nosso metrics.py depois
        from src.modeling.metrics import calcular_metricas
        metricas_treino = calcular_metricas(y_train.values, y_pred, y_pred_proba)

        plots = self.artifacts_cfg.get('plots', [])
        caminhos: dict[str, Path | None] = {}

        if 'confusion_matrix' in plots:
            caminhos['confusion_matrix'] = self.plot_confusion_matrix(y_train, y_pred, model_name)

        if 'roc_curve' in plots and y_pred_proba is not None:
            caminhos['roc_curve'] = self.plot_roc_curve(y_train, y_pred_proba, model_name)

        if 'precision_recall_curve' in plots and y_pred_proba is not None:
            caminhos['precision_recall_curve'] = self.plot_precision_recall_curve(y_train, y_pred_proba, model_name)

        if 'feature_importance' in plots:
            caminhos['feature_importance'] = self.plot_feature_importance(model, list(X_train.columns), X_train, y_train, model_name)

        return caminhos, metricas_treino