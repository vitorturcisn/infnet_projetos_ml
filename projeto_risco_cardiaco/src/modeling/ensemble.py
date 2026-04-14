"""
modeling/ensemble.py — Construtor de Voting Classifier (Ensemble) robusto.
"""
from __future__ import annotations
import logging
from typing import Any
import mlflow
import optuna
from sklearn.ensemble import VotingClassifier
from src.modeling.model_factory import construir_pipeline
from src.modeling.cross_validation import CVRunner
from src.modeling.metrics import agregar_metricas_folds

class EnsembleBuilder:
    def __init__(self, ensembles_cfg: dict, cv_runner: CVRunner, pipe_cfg: dict, feat_red_cfg: dict,
                 n_trials_global: int = 10, seed: int = 42, logger: logging.Logger | None = None) -> None:
        self.ensembles_cfg = ensembles_cfg
        self.cv_runner = cv_runner
        self.pipe_cfg = pipe_cfg
        self.feat_red_cfg = feat_red_cfg
        self.n_trials_global = n_trials_global
        self.seed = seed
        self.logger = logger

    def _construir_estimadores_base(self, top_n_entries: list[tuple]) -> list[tuple]:
        """Reconstrói os pipelines base garantindo a inclusão do PCA/LDA específico de cada um."""
        estimadores = []
        for nome, resultado in top_n_entries:
            pipeline = construir_pipeline(
                model_cfg=resultado['model_cfg'],
                params_modelo=resultado['best_params'],
                params_reducer=resultado.get('reducer_params'), # <── Crucial: Puxa o redutor otimizado
                pipe_cfg=self.pipe_cfg,
                seed=self.seed
            )
            estimadores.append((nome, pipeline))
        return estimadores

    def construir_voting(self, top_n_entries: list[tuple], X_train: Any, y_train: Any) -> dict | None:
        cfg_voting = self.ensembles_cfg.get('voting', {})
        if not cfg_voting.get('enabled', True):
            return None

        n_trials = cfg_voting.get('optuna_trials', self.n_trials_global)
        nomes_base = [nome for nome, _ in top_n_entries]

        if self.logger: self.logger.info('   [OPTUNA] voting    (%d trials) ...', n_trials)

        with mlflow.start_run(run_name='optuna_voting_ensemble', tags={'stage': 'optuna', 'model': 'voting'}):
            def _objetivo_voting(trial: optuna.Trial) -> float:
                # Otimiza os pesos de cada modelo no voto final
                pesos = [trial.suggest_int(f'w_{nome}', 1, 5) for nome in nomes_base]
                
                voting = VotingClassifier(
                    estimators=self._construir_estimadores_base(top_n_entries),
                    voting='soft', # 'soft' é essencial para calcular probabilidades (e Recall)
                    weights=pesos,
                    n_jobs=-1
                )
                
                fold_mets = self.cv_runner.executar(voting, X_train, y_train)
                agg = agregar_metricas_folds(fold_mets)

                # Loga o progresso do ensemble no MLflow
                with mlflow.start_run(run_name=f'voting_trial_{trial.number}', nested=True):
                    mlflow.log_params({f'w_{n}': w for n, w in zip(nomes_base, pesos)})
                    mlflow.log_metric('cv_roc_auc_mean', agg['cv_roc_auc_mean'])
                    mlflow.log_metric('cv_recall_mean', agg.get('cv_recall_mean', 0))
                
                return agg['cv_roc_auc_mean']

            estudo = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
            estudo.optimize(_objetivo_voting, n_trials=n_trials)

        # Treina o modelo final de votação com os melhores pesos encontrados
        melhores_pesos = [estudo.best_params[f'w_{nome}'] for nome in nomes_base]
        melhor_voting = VotingClassifier(
            estimators=self._construir_estimadores_base(top_n_entries),
            voting='soft',
            weights=melhores_pesos,
            n_jobs=-1
        )
        
        fold_mets_voting = self.cv_runner.executar(melhor_voting, X_train, y_train)
        agg_voting = agregar_metricas_folds(fold_mets_voting)

        return {
            **agg_voting,
            'fold_metrics': fold_mets_voting,
            'model_cfg': {'module': 'sklearn.ensemble', 'class': 'VotingClassifier'},
            'best_params': estudo.best_params,
            'reducer_params': {'method': 'ensemble_hybrid'}, # Indica que é uma composição
            'tuned': True,
            '_instance': melhor_voting
        }