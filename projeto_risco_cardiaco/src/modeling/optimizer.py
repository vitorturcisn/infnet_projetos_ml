"""
modeling/optimizer.py — Otimização de Hiperparâmetros (Optuna).
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
    tipo = spec['type']
    if tipo == 'log_float': return trial.suggest_float(nome, float(spec['low']), float(spec['high']), log=True)
    elif tipo == 'float': return trial.suggest_float(nome, float(spec['low']), float(spec['high']))
    elif tipo == 'int': return trial.suggest_int(nome, int(spec['low']), int(spec['high']))
    elif tipo == 'categorical': return trial.suggest_categorical(nome, spec['choices'])
    raise ValueError(f'Tipo {tipo!r} desconhecido')

def _params_reducer_padrao(feat_red_cfg: dict) -> dict:
    return {'method': feat_red_cfg.get('method', 'none')}

def _separar_params_reducer(best_params_all: dict, feat_red_cfg: dict) -> tuple[dict, dict]:
    params_estimador = {k: v for k, v in best_params_all.items() if not k.startswith('reducer_')}
    params_reducer = {'method': best_params_all.get('reducer_method', feat_red_cfg.get('method', 'none'))}
    return params_estimador, params_reducer

class OptunaOptimizer(BaseOptimizer):
    def __init__(self, cfg_optuna: dict, cv_runner: CVRunner, pipe_cfg: dict, seed: int, n_trials_global: int = 10, logger: logging.Logger | None = None) -> None:
        self.cfg_optuna, self.cv_runner, self.pipe_cfg, self.seed, self.n_trials_global, self.logger = cfg_optuna, cv_runner, pipe_cfg, seed, n_trials_global, logger

    def otimizar(self, model_name: str, model_cfg: dict, X_tune: Any, y_tune: Any, pipe_cfg: dict, feat_red_cfg: dict) -> dict:
        n_trials = model_cfg.get('optuna_trials', self.n_trials_global)
        search_space = model_cfg.get('search_space') or {}

        def _objetivo(trial: optuna.Trial) -> float:
            params = {nome: _sugerir_parametro(trial, nome, spec) for nome, spec in search_space.items()}
            pipeline_trial = construir_pipeline(model_cfg, params, _params_reducer_padrao(feat_red_cfg), pipe_cfg)
            
            fold_mets = self.cv_runner.executar(pipeline_trial, X_tune, y_tune)
            agg = agregar_metricas_folds(fold_mets)

            with mlflow.start_run(run_name=f'trial_{trial.number}', nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({
                    'cv_roc_auc_mean': agg['cv_roc_auc_mean'],
                    'cv_f1_mean': agg.get('cv_f1_mean', 0.0),
                })
            # Na classificação, queremos MAXIMIZAR o ROC-AUC
            return agg['cv_roc_auc_mean']

        # IMPORTANTE: direction='maximize' para ROC-AUC
        study = optuna.create_study(direction='maximize', study_name=model_name, sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(_objetivo, n_trials=n_trials, show_progress_bar=False, catch=(Exception,))

        try:
            _ = study.best_value
        except ValueError:
            return {}

        params_est, params_red = _separar_params_reducer(study.best_params, feat_red_cfg)
        return {'estimator_params': params_est, 'reducer_params': params_red, 'study': study}

class OptimizerFactory:
    @staticmethod
    def criar(cfg_modelagem: dict, cv_runner: CVRunner, pipe_cfg: dict, seed: int, logger: logging.Logger | None = None) -> BaseOptimizer:
        cfg_opt = cfg_modelagem.get('optimizer', {})
        n_trials_global = cfg_opt.get('optuna', {}).get('default_trials', 10)
        return OptunaOptimizer(cfg_optuna=cfg_opt.get('optuna', {}), cv_runner=cv_runner, pipe_cfg=pipe_cfg, seed=seed, n_trials_global=n_trials_global, logger=logger)