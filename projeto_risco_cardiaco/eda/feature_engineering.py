import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- Bootstrap igual ao do professor --
def _bootstrap_src_path(base_dir: Path) -> None:
    src_path = str(base_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

def run(config: dict, base_dir: Path) -> pd.DataFrame:
    _bootstrap_src_path(base_dir)
    from src.utils.logger import get_logger
    logger = get_logger("eda.feature_engineering", {"level": "INFO"})

    logger.info("=== Feature Engineering: Risco Cardíaco — INÍCIO ===")
    
    # 1. Carregar dados
    df = pd.read_parquet(base_dir / "data/processed/heart_disease.parquet")
    fe_cfg = config.get("feature_engineering", {})
    target = config.get("target", "Heart Disease Status")
    
    # 2. Criar Razões (Ratios)
    for feat, spec in fe_cfg.get("ratio_features", {}).items():
        df[feat] = df[spec['numerator']] / df[spec['denominator']].replace(0, np.nan)
        logger.info(f"Criada razão: {feat}")

    # 3. Aplicar Log (Para reduzir assimetria)
    for feat in fe_cfg.get("log_features", []):
        df[f"log_{feat}"] = np.log1p(df[feat].clip(lower=0))
        logger.info(f"Aplicado Log1p em: {feat}")

    # 4. Codificação Categórica (Sim/Não para 1/0)
    # Transforma 'Gender', 'Smoking', 'Diabetes' etc em números
    cat_cols = config.get("features", {}).get("categorical", [])
    for col in cat_cols:
        if col in df.columns:
            df[f"{col}_encoded"] = pd.factorize(df[col])[0]
    
    # 5. Análise de Correlação (Ignorando textos automaticamente)
    df['target_numeric'] = df[target].map({'Yes': 1, 'No': 0})
    # O parâmetro numeric_only=True evita o erro com strings
    correlations = df.corr(numeric_only=True)['target_numeric'].sort_values(ascending=False)
    
    # Salvar resultados
    stats_dir = base_dir / "outputs/stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(stats_dir / "20_engineered_feature_correlations.csv")
    
    # 6. Salvar Amostra Enriquecida (Igual ao professor)
    tables_dir = base_dir / "outputs/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.head(1000).to_csv(tables_dir / "09_enriched_dataset_sample.csv", index=False)

    logger.info(f"=== Feature Engineering Concluída! Novas colunas: {df.shape[1] - 21} ===")
    return df