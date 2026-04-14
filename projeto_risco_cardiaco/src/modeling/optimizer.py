"""
modeling/optimizer.py — Otimização de Hiperparâmetros (Optuna) com Redução de Dimensionalidade.
"""
from __future__ import annotations
import logging
from typing import Any
import mlflow
import optuna
from src.modeling.base import BaseOptimizer
from src.modeling.metrics import agregar_metricas_folds
from src.modeling.model_factory import construir_pipeline
from src.modeling.cross_validation import CVRunner

def _sugerir_parametro(trial: optuna.Trial, nome: str, spec: dict) -> Any:
    """Extrai sugestões do Optuna baseadas no tipo definido no YAML."""
    tipo = spec['type']
    if tipo == 'log_float': 
        return trial.suggest_float(nome, float(spec['low']), float(spec['high']), log=True)
    elif tipo == 'float': 
        return trial.suggest_float(nome, float(spec['low']), float(spec['high']))
    elif tipo == 'int': 
        return trial.suggest_int(nome, int(spec['low']), int(spec['high']))
    elif tipo == 'categorical': 
        return trial.suggest_categorical(nome, spec['choices'])
    raise ValueError(f'Tipo {tipo!r} desconhecido')

def _params_reducer_padrao(feat_red_cfg: dict) -> dict:
    return {'method': feat_red_cfg.get('method', 'none')}

class OptunaOptimizer(BaseOptimizer):
    def __init__(self, cfg_optuna: dict, cv_runner: CVRunner, pipe_cfg: dict, seed: int, n_trials_global: int = 10, logger: logging.Logger | None = None) -> None:
        self.cfg_optuna = cfg_optuna
        self.cv_runner = cv_runner
        self.pipe_cfg = pipe_cfg
        self.seed = seed
        self.n_trials_global = n_trials_global
        self.logger = logger

    def otimizar(self, model_name: str, model_cfg: dict, X_tune: Any, y_tune: Any, pipe_cfg: dict, feat_red_cfg: dict) -> dict:
        n_trials = model_cfg.get('optuna_trials', self.n_trials_global)
        search_space_model = model_cfg.get('search_space') or {}
        search_space_red = feat_red_cfg.get('search_space') or {}

        def _objetivo(trial: optuna.Trial) -> float:
            # 1. Sugestões para o MODELO
            params_estimador = {nome: _sugerir_parametro(trial, nome, spec) for nome, spec in search_space_model.items()}
            
            # 2. Sugestões para a REDUÇÃO (O que faltava!)
            params_reducer = {
                'method': trial.suggest_categorical('method', search_space_red.get('method', {}).get('choices', ['none'])),
                'n_components': trial.suggest_int('n_components', 
                                                 search_space_red.get('n_components', {}).get('low', 2), 
                                                 search_space_red.get('n_components', {}).get('high', 7))
            }

            # 3. Construção do Pipeline com AMBOS os parâmetros
            pipeline_trial = construir_pipeline(model_cfg, params_estimador, params_reducer, pipe_cfg)
            
            try:
                fold_mets = self.cv_runner.executar(pipeline_trial, X_tune, y_tune)
                agg = agregar_metricas_folds(fold_mets)

                # LOG NO MLFLOW: Agora logamos tudo (Modelo + Redução) para aparecer na tabela
                with mlflow.start_run(run_name=f'trial_{trial.number}', nested=True):
                    mlflow.log_params(params_estimador)
                    mlflow.log_params(params_reducer) # <── AGORA APARECE NO MLFLOW!
                    mlflow.log_metrics({
                        'cv_roc_auc_mean': agg['cv_roc_auc_mean'],
                        'recall': agg.get('cv_recall_mean', agg.get('recall', 0.0)),
                        'f1': agg.get('cv_f1_mean', agg.get('f1', 0.0))
                    })
                
                return agg['cv_roc_auc_mean']
            except Exception as e:
                if self.logger: self.logger.error(f"Erro no trial {trial.number}: {e}")
                return 0.0

        # Direção MAXIMIZE para ROC-AUC ou RECALL
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(_objetivo, n_trials=n_trials, catch=(Exception,))

        try:
            melhor_trial = study.best_trial
        except ValueError:
            return {}

        # Separa os parâmetros para retornar ao step.py
        melhores_params_todos = melhor_trial.params
        params_est = {k: v for k, v in melhores_params_todos.items() if k in search_space_model}
        params_red = {
            'method': melhores_params_todos.get('method', feat_red_cfg.get('method', 'none')),
            'n_components': melhores_params_todos.get('n_components', 5)
        }

        return {
            'estimator_params': params_est, 
            'reducer_params': params_red, 
            'value': study.best_value
        }

class OptimizerFactory:
    @staticmethod
    def criar(cfg_modelagem: dict, cv_runner: CVRunner, pipe_cfg: dict, seed: int, logger: logging.Logger | None = None) -> BaseOptimizer:
        cfg_opt = cfg_modelagem.get('optimizer', {})
        n_trials_global = cfg_opt.get('optuna', {}).get('default_trials', 10)
        return OptunaOptimizer(
            cfg_optuna=cfg_opt.get('optuna', {}), 
            cv_runner=cv_runner, 
            pipe_cfg=pipe_cfg, 
            seed=seed, 
            n_trials_global=n_trials_global, 
            logger=logger
        )