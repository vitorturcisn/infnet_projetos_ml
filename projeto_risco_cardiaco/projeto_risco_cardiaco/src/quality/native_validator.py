"""
src/quality/native_validator.py — Validador Nativo (Pandas)
Substituto direto para o Great Expectations em ambientes incompatíveis.
Lê o mesmo YAML e executa as validações nativamente.
"""
from __future__ import annotations
import logging
from typing import Any
import pandas as pd
from src.quality.base import QualityValidator

class NativePandasValidator(QualityValidator):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger)

    def validate(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        quality_cfg = config.get("quality", {})
        fail_on_error = quality_cfg.get("fail_pipeline_on_error", True)
        
        self._logger.info("Iniciando Validador Nativo Pandas (Bypass GE)...")
        results_list = []
        passed = 0
        failed = 0

        # 1. Validar Expectativas de Tabela
        for exp in config.get("table_expectations", []):
            sucesso, detalhes = self._evaluate_table_exp(df, exp)
            if sucesso: passed += 1
            else: failed += 1
            results_list.append(self._format_result(exp, None, sucesso, detalhes))

        # 2. Validar Expectativas de Coluna
        for coluna, exps in config.get("column_expectations", {}).items():
            for exp in exps:
                sucesso, detalhes = self._evaluate_column_exp(df, coluna, exp)
                if sucesso: passed += 1
                else: failed += 1
                results_list.append(self._format_result(exp, coluna, sucesso, detalhes))

        total = passed + failed
        success = failed == 0

        self._logger.info("-" * 60)
        self._logger.info("Resultados Nativos: %d Passou | %d Falhou", passed, failed)
        self._logger.info("-" * 60)

        if fail_on_error and not success:
            raise RuntimeError(f"Qualidade REPROVADA: {failed}/{total} checks falharam.")

        return {
            "success": success,
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": {"results": results_list} # Formato esperado pelo report_writer
        }

    def _evaluate_table_exp(self, df: pd.DataFrame, exp: dict) -> tuple[bool, dict]:
        tipo = exp["type"]
        kwargs = exp.get("kwargs", {})
        
        if tipo == "expect_table_row_count_to_be_between":
            count = len(df)
            sucesso = kwargs.get("min_value", 0) <= count <= kwargs.get("max_value", float('inf'))
            return sucesso, {"observed_value": count}
            
        elif tipo == "expect_table_columns_to_match_set":
            esperado = set(kwargs.get("column_set", []))
            observado = set(df.columns)
            sucesso = esperado == observado
            return sucesso, {"missing": list(esperado - observado), "unexpected": list(observado - esperado)}
            
        return False, {"error": "Expectation não implementada no validador nativo"}

    def _evaluate_column_exp(self, df: pd.DataFrame, col: str, exp: dict) -> tuple[bool, dict]:
        if col not in df.columns:
            return False, {"error": f"Coluna {col} não existe"}

        tipo = exp["type"]
        kwargs = exp.get("kwargs", {})
        mostly = kwargs.get("mostly", 1.0)
        serie = df[col]
        serie_valida = serie.dropna()

        if tipo == "expect_column_values_to_not_be_null":
            taxa_preenchimento = 1.0 - (serie.isnull().sum() / len(serie))
            return taxa_preenchimento >= mostly, {"fill_rate": taxa_preenchimento}

        if len(serie_valida) == 0:
            return False, {"error": "Coluna totalmente vazia"}

        if tipo == "expect_column_values_to_be_between":
            min_v = kwargs.get("min_value", float('-inf'))
            max_v = kwargs.get("max_value", float('inf'))
            taxa_sucesso = serie_valida.between(min_v, max_v).mean()
            return taxa_sucesso >= mostly, {"success_rate": taxa_sucesso, "min_obs": serie_valida.min(), "max_obs": serie_valida.max()}

        elif tipo == "expect_column_values_to_be_in_set":
            valores_permitidos = set(kwargs.get("value_set", []))
            taxa_sucesso = serie_valida.isin(valores_permitidos).mean()
            return taxa_sucesso >= mostly, {"success_rate": taxa_sucesso, "unique_obs": list(serie_valida.unique())[:10]}

        return False, {"error": "Expectation não implementada no validador nativo"}

    def _format_result(self, exp, col, success, details):
        return {
            "success": success,
            "expectation_config": {"type": exp["type"], "kwargs": {**exp.get("kwargs", {}), "column": col}},
            "result": details
        }