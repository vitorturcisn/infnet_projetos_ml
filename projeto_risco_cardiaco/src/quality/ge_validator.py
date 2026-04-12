"""
src/quality/ge_validator.py — Implementação para Great Expectations v0.18.22.
"""
from __future__ import annotations
import logging
from typing import Any
import pandas as pd
from src.quality.base import ExpectationResolver, QualityValidator

class GreatExpectationsValidator(QualityValidator):
    def __init__(self, resolver: ExpectationResolver, logger: logging.Logger, gx) -> None:
        super().__init__(logger)
        self._resolver = resolver
        self._gx = gx

    def validate(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        quality_cfg = config.get("quality", {})
        fail_on_error = quality_cfg.get("fail_pipeline_on_error", True)

        # Na v0.18, o Pandas Dataset já tem os métodos acoplados nativamente
        ge_df = self._gx.from_pandas(df)

        self._logger.info("Carregando regras do quality.yaml...")
        
        # Aplica regras de tabela
        for exp in config.get("table_expectations", []):
            getattr(ge_df, exp["type"])(**exp.get("kwargs", {}))

        # Aplica regras de coluna
        for coluna, exps in config.get("column_expectations", {}).items():
            for exp in exps:
                getattr(ge_df, exp["type"])(column=coluna, **exp.get("kwargs", {}))

        self._logger.info("Executando validação de qualidade (GE v0.18)...")
        results = ge_df.validate()

        total = results["statistics"]["evaluated_expectations"]
        passed = results["statistics"]["successful_expectations"]
        failed = results["statistics"]["unsuccessful_expectations"]

        self._logger.info("-" * 60)
        self._logger.info("Resultados: %d Passou | %d Falhou", passed, failed)
        self._logger.info("-" * 60)

        if fail_on_error and not results["success"]:
            raise RuntimeError(
                f"Qualidade de dados REPROVADA: {failed}/{total} checks falharam.\n"
                "Para continuar, altere fail_pipeline_on_error para false no YAML."
            )

        return {
            "success": results["success"],
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": results  # Na v0.18, retorna um dicionário JSON-friendly
        }