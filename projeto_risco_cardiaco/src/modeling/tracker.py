"""
modeling/tracker.py — Rastreador MLflow atualizado para capturar PCA/LDA e métricas clínicas.
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
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.root_dir = root_dir
        self.logger = logger
        self._configurar()

    def _configurar(self) -> None:
        uri = self.tracking_uri
        if not uri.startswith('sqlite:') and not uri.startswith('http'):
            uri = (self.root_dir / uri).as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.experiment_name)

    def contexto_otimizacao(self, model_name: str, stage: str = 'optuna'):
        """Cria um run pai para agrupar as tentativas (trials) do Optuna."""
        return mlflow.start_run(run_name=f'optuna_{model_name}', tags={'stage': stage, 'model': model_name})

    def logar_melhor_modelo(self, model_name: str, model: Any, best_params: dict, reducer_params: dict, 
                            cv_metrics: dict, train_metrics: dict, fold_metrics: list[dict], 
                            plot_paths: list[str], tuned: bool) -> str:
        """
        Loga o modelo campeão com todos os detalhes de engenharia.
        Essencial para a Parte 5 e 6 do projeto.
        """
        with mlflow.start_run(run_name=f'best_model_{model_name}', tags={'stage': 'production', 'model': model_name}, nested=True) as run:
            # 1. Log de Parâmetros (Unindo Estimador + Redutor)
            mlflow.log_params({f"model_{k}": v for k, v in best_params.items()})
            mlflow.log_params({f"reducer_{k}": v for k, v in reducer_params.items()})
            
            # 2. Log de Métricas (Foco em Recall e ROC-AUC)
            mlflow.log_metrics(cv_metrics)
            mlflow.log_metrics(train_metrics)
            
            # 3. Log de Artefatos (Plots de diagnósticos)
            for caminho in plot_paths:
                if caminho and Path(caminho).exists():
                    mlflow.log_artifact(str(caminho), artifact_path='plots')
            
            # 4. Persistência do Modelo (Pipeline Completo)
            mlflow.sklearn.log_model(model, artifact_path='model')
            
            return run.info.run_id

    def logar_holdout(self, run_id: str, holdout_metrics: dict, step: int = 0) -> None:
        """Loga a performance em dados nunca vistos (Teste de Sanidade)."""
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f'holdout_{k}': v for k, v in holdout_metrics.items()})

    def registrar_modelo(self, run_id: str, registry_name: str) -> None:
        """Envia o modelo para o Model Registry (Versionamento V1, V2...)."""
        model_uri = f'runs:/{run_id}/model'
        try:
            mlflow.register_model(model_uri=model_uri, name=registry_name)
        except Exception as e:
            if self.logger: self.logger.warning(f"Falha ao registrar modelo: {e}")

    def salvar_resumo_json(self, output_dir: Path, best_model_name: str, best_run_id: str, 
                          best_result: dict, holdout_metrics: dict, top_n_names: list[str], 
                          full_ranking_records: list[dict]) -> Path:
        """Gera o relatório final em JSON para documentação do projeto."""
        resumo = {
            'projeto': 'Risco Cardíaco - Triagem Agressiva',
            'best_model': best_model_name,
            'best_run_id': best_run_id,
            'cv_roc_auc_mean': round(best_result['cv_roc_auc_mean'], 4),
            'holdout_roc_auc': round(holdout_metrics.get('roc_auc', 0), 4),
            'holdout_recall': round(holdout_metrics.get('recall', 0), 4),
            'reducer_used': best_result.get('reducer_params', {}).get('method', 'none'),
            'best_params': best_result.get('best_params', {}),
            'top_models': top_n_names,
            'ranking': full_ranking_records
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        caminho = output_dir / 'experiment_summary.json'
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)
        return caminho