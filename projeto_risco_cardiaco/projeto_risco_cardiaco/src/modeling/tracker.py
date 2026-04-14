"""
modeling/tracker.py — Rastreador MLflow para experimentação MLOps.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
import mlflow
import mlflow.sklearn

class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str, root_dir: Path, logger: logging.Logger | None = None) -> None:
        self.tracking_uri, self.experiment_name, self.root_dir, self.logger = tracking_uri, experiment_name, root_dir, logger
        self._configurar()

    def _configurar(self) -> None:
        uri = self.tracking_uri
        if not uri.startswith('sqlite:') and not uri.startswith('http'):
            uri = (self.root_dir / uri).as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.experiment_name)

    def logar_baseline(self, model_name: str, params: dict, fold_metrics: list[dict], agg_metrics: dict, training_time: float, model_class: str, reducer_method: str) -> None:
        with mlflow.start_run(run_name=f'baseline_{model_name}', tags={'stage': 'baseline', 'model': model_name}):
            mlflow.log_params({str(k): (str(v) if v is None else v) for k, v in params.items()})
            mlflow.log_metrics(agg_metrics)

    def contexto_otimizacao(self, model_name: str, stage: str = 'optuna'):
        return mlflow.start_run(run_name=f'optuna_{model_name}', tags={'stage': stage, 'model': model_name})

    def logar_melhor_optuna(self, best_params: dict, best_cv_rmse: float, n_trials: int, study: Any | None = None, artifact_paths: list[Path] | None = None) -> None:
        mlflow.log_params({f'best_{k}': str(v) for k, v in best_params.items()})
        mlflow.log_metrics({'best_cv_roc_auc': best_cv_rmse})

    def logar_melhor_modelo(self, model_name: str, model: Any, best_params: dict, reducer_params: dict, cv_metrics: dict, train_metrics: dict, fold_metrics: list[dict], plot_paths: dict, tuned: bool) -> str:
        with mlflow.start_run(run_name=f'best_model_{model_name}', tags={'stage': 'best_model'}) as run:
            mlflow.log_params({k: str(v) for k, v in best_params.items()})
            mlflow.log_metrics(cv_metrics)
            mlflow.log_metrics(train_metrics)
            for caminho in plot_paths.values():
                if caminho and Path(caminho).exists():
                    mlflow.log_artifact(str(caminho), artifact_path='plots')
            mlflow.sklearn.log_model(model, artifact_path='model')
            return run.info.run_id

    def logar_holdout(self, run_id: str, holdout_metrics: dict, delta_pct: float, holdout_plot_path: Path | None = None) -> None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                'holdout_roc_auc': holdout_metrics.get('roc_auc', 0),
                'holdout_f1': holdout_metrics.get('f1', 0),
                'holdout_recall': holdout_metrics.get('recall', 0),
            })

    def registrar_modelo(self, run_id: str, registry_name: str) -> None:
        try:
            mlflow.register_model(model_uri=f'runs:/{run_id}/model', name=registry_name)
        except Exception:
            pass

    def salvar_resumo_json(self, output_dir: Path, best_model_name: str, best_run_id: str, best_result: dict, holdout_metrics: dict, top_n_names: list[str], full_ranking_records: list[dict]) -> Path:
        resumo = {
            'best_model': best_model_name,
            'best_run_id': best_run_id,
            'cv_roc_auc_mean': round(best_result['cv_roc_auc_mean'], 4),
            'cv_roc_auc_std': round(best_result['cv_roc_auc_std'], 4),
            'holdout_roc_auc': round(holdout_metrics.get('roc_auc', 0), 4),
            'holdout_recall': round(holdout_metrics.get('recall', 0), 4),
            'best_params': {k: str(v) for k, v in best_result.get('best_params', {}).items()},
            'top_base_models': top_n_names,
            'all_models_ranked': full_ranking_records,
        }
        caminho = output_dir / 'experiment_summary.json'
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)
        return caminho