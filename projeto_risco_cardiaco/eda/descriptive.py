import pandas as pd
import json
from pathlib import Path

def run(config: dict, base_dir: Path):
    from src.utils.logger import get_logger
    logger = get_logger("eda.descriptive", {"level": "INFO"})

    # 1. Caminhos
    input_path = base_dir / "data/processed/heart_disease.parquet"
    output_dir = base_dir / "outputs/stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Carga
    df = pd.read_parquet(input_path)
    
    # 3. Basic Info (Igual ao 01_basic_info.json do professor)
    basic_info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage().sum() / 1024**2, 2),
        "missing_values": df.isnull().sum().to_dict()
    }
    with open(output_dir / "01_basic_info.json", "w") as f:
        json.dump(basic_info, f, indent=2)

    # 4. Estatísticas Descritivas (02_descriptive_stats.csv)
    # Seleciona apenas o que é número
    stats = df.select_dtypes(include=['float64', 'int64']).describe().T
    stats['skewness'] = df.select_dtypes(include=['float64', 'int64']).skew()
    stats.to_csv(output_dir / "02_descriptive_stats.csv")

    logger.info("Relatórios de estatística descritiva gerados com sucesso.")