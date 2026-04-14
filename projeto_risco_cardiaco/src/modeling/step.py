"""
modeling/step.py — Etapa de Modelagem do Pipeline MLOps.
Versão atualizada para suporte a PCA/LDA e rastreamento completo no MLflow.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import optuna
from sklearn.model_selection import train_test_split

from src.core.base import PipelineStep
from src.utils.config_loader import load_yaml
from src.modeling.model_factory import construir_pipeline
from src.modeling.cross_validation import CVRunner
from src.modeling.optimizer import OptimizerFactory, _params_reducer_padrao
from src.modeling.ensemble import EnsembleBuilder
from src.modeling.evaluator import HoldoutEvaluator
from src.modeling.artifacts import ArtifactGenerator
from src.modeling.tracker import MLflowTracker
from src.modeling.metrics import agregar_metricas_folds

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelingStep(PipelineStep):
    def __init__(self, context: Any) -> None:
        super().__init__(logger=context.logger)
        self.context = context
        # Carrega a configuração do modeling.yaml (incluindo o feature_reduction)
        self._cfg = {**self.context.pipeline_cfg, **load_yaml(self.context.config_dir / 'modeling.yaml')}

    def run(self) -> None:
        self.logger.info('=== Modelagem e Experimentação — Risco Cardíaco ===')

        X, y = self._carregar_features()
        X_train, X_holdout, y_train, y_holdout = self._dividir_treino_holdout(X, y)

        cv_runner = CVRunner.de_config(self._cfg.get('cv', {}), self._cfg.get('modeling', {}).get('random_seed', 42))
        tracker = MLflowTracker(
            tracking_uri=self._cfg.get('modeling', {}).get('tracking_uri', 'sqlite:///mlruns.db'),
            experiment_name=self._cfg.get('modeling', {}).get('experiment_name', 'cardiac-risk-arena-recall'),
            root_dir=self.context.root_dir,
            logger=self.logger,
        )
        
        otimizador = OptimizerFactory.criar(
            self._cfg, cv_runner, self._cfg.get('pipeline', {}), 
            self._cfg.get('modeling', {}).get('random_seed', 42), self.logger
        )
        
        gerador_artefatos = ArtifactGenerator(
            self.context.root_dir / self._cfg.get('artifacts', {}).get('output_dir', 'outputs/modeling'), 
            self._cfg.get('artifacts', {}), self.logger
        )
        avaliador = HoldoutEvaluator(logger=self.logger)

        # ── Execução das Seções ──
        todos_resultados = self._executar_baseline(cv_runner, tracker, X_train, y_train)
        todos_resultados = self._executar_otimizacao(otimizador, tracker, cv_runner, todos_resultados, X_train, y_train)
        todos_resultados = self._executar_ensembles(todos_resultados, cv_runner, X_train, y_train)
        
        # Seleção do Campeão
        nome_melhor, resultado_melhor = self._selecionar_melhor(todos_resultados)

        # Treinamento Final (Garante uso dos parâmetros de redução descobertos)
        melhor_modelo = self._treinar_melhor_modelo(nome_melhor, resultado_melhor)
        melhor_modelo.fit(X_train, y_train)

        # Geração de Diagnósticos (Plots)
        plot_paths, metricas_treino = gerador_artefatos.gerar_diagnosticos_modelo(
            melhor_modelo, nome_melhor, X_train, y_train, resultado_melhor['fold_metrics']
        )

        # Log Final no MLflow (Agora com reducer_params dinâmicos)
        best_run_id = tracker.logar_melhor_modelo(
            model_name=nome_melhor, 
            model=melhor_modelo, 
            best_params=resultado_melhor['best_params'],
            reducer_params=resultado_melhor.get('reducer_params', {}), # <── FIX: Passa PCA/LDA para o MLflow
            cv_metrics={'cv_roc_auc_mean': resultado_melhor['cv_roc_auc_mean']},
            train_metrics=metricas_treino, 
            fold_metrics=resultado_melhor['fold_metrics'],
            plot_paths=plot_paths, 
            tuned=resultado_melhor.get('tuned', False)
        )

        # Avaliação de Robustez (Holdout)
        metricas_holdout = avaliador.avaliar(melhor_modelo, X_holdout, y_holdout)
        avaliador.diagnosticar_robustez(resultado_melhor['cv_roc_auc_mean'], metricas_holdout['roc_auc'])

        tracker.logar_holdout(best_run_id, metricas_holdout, 0)
        tracker.registrar_modelo(best_run_id, self._cfg.get('modeling', {}).get('registry_name', 'cardiac-risk-best'))

        # Resumo final
        ranking = sorted(todos_resultados.items(), key=lambda x: x[1]['cv_roc_auc_mean'], reverse=True)
        tracker.salvar_resumo_json(
            self.context.root_dir / self._cfg.get('artifacts', {}).get('output_dir', 'outputs/modeling'),
            nome_melhor, best_run_id, resultado_melhor, metricas_holdout,
            [nome for nome, _ in ranking[:2]], [{'modelo': k, 'cv_roc_auc': v['cv_roc_auc_mean']} for k, v in ranking]
        )

        self.logger.info('═' * 60)
        self.logger.info('=== Modelagem CONCLUÍDA ===')
        self.logger.info('Melhor modelo : %s', nome_melhor)
        self.logger.info('  CV ROC-AUC  : %.4f', resultado_melhor['cv_roc_auc_mean'])
        self.logger.info('  Holdout AUC : %.4f', metricas_holdout.get('roc_auc', 0))
        self.logger.info('═' * 60)

    # ── Métodos Auxiliares ───────────────────────────────────────────────────

    def _carregar_features(self):
        features_file = self.context.root_dir / 'data/features/heart_disease_features.parquet'
        df = pq.read_table(str(features_file)).to_pandas()
        X = df.drop(columns=['target_numeric', 'Heart Disease Status'], errors='ignore')
        y = df['target_numeric']
        return X, y

    def _dividir_treino_holdout(self, X, y):
        test_size = self._cfg.get('holdout', {}).get('test_size', 0.2)
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        return X_train, X_holdout, y_train, y_holdout

    def _executar_baseline(self, cv_runner, tracker, X_train, y_train):
        self.logger.info('SEÇÃO 3: Baseline CV')
        resultados = {}
        for nome_modelo, cfg_modelo in self._cfg.get('models', {}).items():
            if not cfg_modelo.get('enabled', True): continue
            
            # Baseline não usa redução de dimensionalidade (None)
            pipeline = construir_pipeline(cfg_modelo, None, None, self._cfg.get('pipeline', {}))
            fold_mets = cv_runner.executar(pipeline, X_train, y_train)
            agg = agregar_metricas_folds(fold_mets)
            
            resultados[nome_modelo] = {
                **agg, 
                'fold_metrics': fold_mets, 
                'model_cfg': cfg_modelo, 
                'best_params': dict(cfg_modelo.get('default_params') or {}),
                'reducer_params': {'method': 'none'}, # <── Registro explícito
                'tuned': False
            }
            self.logger.info('    %s Baseline AUC: %.4f', nome_modelo, agg['cv_roc_auc_mean'])
        return resultados

    def _executar_otimizacao(self, otimizador, tracker, cv_runner, resultados, X_train, y_train):
        self.logger.info('SEÇÃO 4: Otimização Optuna')
        for nome_modelo, cfg_modelo in self._cfg.get('models', {}).items():
            if not cfg_modelo.get('enabled', True) or not cfg_modelo.get('search_space'): continue
            self.logger.info('  Otimizando %s...', nome_modelo)
            
            with tracker.contexto_otimizacao(nome_modelo):
                res_opt = otimizador.otimizar(
                    nome_modelo, cfg_modelo, X_train, y_train, 
                    self._cfg.get('pipeline', {}), self._cfg.get('feature_reduction', {})
                )
            
            if not res_opt: continue
            
            # Reconstrói pipeline com os melhores parâmetros de redução (PCA/LDA) e estimador
            pipeline_otimizado = construir_pipeline(
                cfg_modelo, 
                res_opt['estimator_params'], 
                res_opt['reducer_params'], # <── FIX: Agora passa o redutor escolhido pelo Optuna
                self._cfg.get('pipeline', {})
            )
            
            fold_mets = cv_runner.executar(pipeline_otimizado, X_train, y_train)
            agg = agregar_metricas_folds(fold_mets)
            
            # Atualiza se houver ganho de performance
            if agg['cv_roc_auc_mean'] > resultados[nome_modelo]['cv_roc_auc_mean']:
                resultados[nome_modelo].update({
                    **agg, 
                    'fold_metrics': fold_mets, 
                    'best_params': res_opt['estimator_params'],
                    'reducer_params': res_opt['reducer_params'], # <── FIX: Salva os parâmetros de redução
                    'tuned': True
                })
                self.logger.info('    ✔ Melhorou para AUC: %.4f (Redutor: %s)', 
                                 agg['cv_roc_auc_mean'], res_opt['reducer_params'].get('method'))
        return resultados

    def _executar_ensembles(self, resultados, cv_runner, X_train, y_train):
        builder = EnsembleBuilder(
            self._cfg.get('ensembles', {}), cv_runner, 
            self._cfg.get('pipeline', {}), self._cfg.get('feature_reduction', {}), 
            logger=self.logger
        )
        top_entries = sorted(
            resultados.items(), key=lambda x: x[1]['cv_roc_auc_mean'], reverse=True
        )[:self._cfg.get('ensembles', {}).get('top_n_base_models', 2)]
        
        res_voting = builder.construir_voting(top_entries, X_train, y_train)
        if res_voting: resultados['voting'] = res_voting
        return resultados

    def _selecionar_melhor(self, resultados):
        ranking = sorted(resultados.items(), key=lambda x: x[1]['cv_roc_auc_mean'], reverse=True)
        return ranking[0]

    def _treinar_melhor_modelo(self, nome, resultado):
        if '_instance' in resultado: return resultado['_instance']
        
        # Reconstrói o pipeline final com PCA/LDA integrado se necessário
        return construir_pipeline(
            resultado['model_cfg'], 
            resultado['best_params'], 
            resultado.get('reducer_params'), # <── FIX: Garante que o modelo final use o redutor correto
            self._cfg.get('pipeline', {})
        )