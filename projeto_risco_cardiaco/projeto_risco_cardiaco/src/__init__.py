"""Pacote raiz src — exporta a API pública do pipeline."""

# 1. Core (Orquestração)
from src.core.context import PipelineContext

# 2. Ingestão
from src.ingestion.downloader import KaggleDownloader
from src.ingestion.parquet_writer import CsvToParquetIngester

# 3. Qualidade (Mantenha comentado se ainda não criou os arquivos de Quality)
from src.quality import (
     GeExpectationResolver,
     GreatExpectationsValidator,
     QualityReportWriter,
     QualityValidator,
)

# 4. Pré-processamento (DESCOMENTADO AGORA ✅)
from src.preprocessing import (
    PreprocessingStep,
    PreprocessingPipelineBuilder,
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    PolynomialFeatureTransformer,
    CardiacCategoricalEncoder,
    FeatureSelector,
)

__all__ = [
    # Orquestração
    "PipelineContext",
    # Ingestão
    "KaggleDownloader",
    "CsvToParquetIngester",
    # Pré-processamento
    "PreprocessingStep",
    "PreprocessingPipelineBuilder",
    "BinaryFlagTransformer",
    "RatioFeatureTransformer",
    "LogTransformer",
    "PolynomialFeatureTransformer",
    "CardiacCategoricalEncoder",
    "FeatureSelector",
]