"""
src/quality/report_writer.py — Serialização do relatório de qualidade em JSON.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from src.quality.base import QualityReportWriterBase

class QualityReportWriter(QualityReportWriterBase):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger)

    def write(self, summary: dict[str, Any], output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"quality_report_{timestamp}.json"

        report = {
            "success": summary["success"],
            "total": summary["total"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "details": self._extrair_detalhes(summary["results"]),
        }

        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

        self._logger.info("Relatório JSON salvo em: %s", report_path)
        return report_path

    @staticmethod
    def _extrair_detalhes(validation_result) -> list[dict[str, Any]]:
        detalhes = []
        for r in validation_result.get("results", []):
            exp_config = r.get("expectation_config", {})
            kwargs = dict(exp_config.get("kwargs", {}))
            coluna = kwargs.pop("column", None)

            detalhes.append({
                "type": exp_config.get("expectation_type"),
                "column": coluna,
                "success": r.get("success"),
                "kwargs": kwargs,
                "result": r.get("result", {})
            })
        return detalhes