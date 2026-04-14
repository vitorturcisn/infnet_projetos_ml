"""
src/quality/step.py — Etapa de Qualidade do Pipeline.
"""
from __future__ import annotations
import pandas as pd

try:
    from src.core.base import PipelineStep
except ImportError:
    PipelineStep = object

from src.utils.config_loader import load_yaml
# Usando nosso novo validador nativo que bypassa o GE!
from src.quality.native_validator import NativePandasValidator
from src.quality.report_writer import QualityReportWriter

class QualityStep(PipelineStep):
    def __init__(self, context) -> None:
        if PipelineStep is not object:
            super().__init__(logger=context.logger)
        self.logger = context.logger
        self.context = context
        self._config = load_yaml(self.context.config_dir / "quality.yaml")

    def run(self) -> None:
        self.logger.info("=" * 60)
        self.logger.info("=== Data Quality Gate (Native Pandas Engine) ===")
        
        caminho_entrada = self.context.root_dir / "data/processed/heart_disease.parquet"
        self.logger.info("Carregando dados: %s", caminho_entrada)
        df = pd.read_parquet(caminho_entrada)

        # Injetando a nova dependência!
        validator = NativePandasValidator(self.logger)
        writer = QualityReportWriter(self.logger)

        # A execução continua idêntica
        resumo = validator.validate(df, self._config)

        output_dir = self.context.root_dir / self._config["quality"]["output_dir"]
        writer.write(resumo, output_dir)
        
        self.logger.info("=" * 60)