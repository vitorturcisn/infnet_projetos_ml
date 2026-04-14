import sys
from pathlib import Path
import pandas as pd
import numpy as np

# -- Bootstrap de Caminhos --
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE / "src") not in sys.path:
    sys.path.insert(0, str(_BASE / "src"))

def _add_binned_columns(df, config, logger):
    """Agrupa dados contínuos em categorias (Ex: Idade -> Faixa Etária)."""
    df = df.copy()
    bins_cfg = config.get("pivot_tables", {}).get("bins", {})
    
    bin_map = {
        "age_category": ("Age", "age"),
        "bmi_category": ("BMI", "bmi"),
        "bp_category": ("Blood Pressure", "blood_pressure")
    }

    for new_col, (src_col, cfg_key) in bin_map.items():
        if src_col in df.columns and cfg_key in bins_cfg:
            df[new_col] = pd.cut(df[src_col], bins=bins_cfg[cfg_key]["cuts"], 
                                 labels=bins_cfg[cfg_key]["labels"], include_lowest=True)
            logger.info(f"Criada categoria: {new_col}")
    return df

def run(config: dict, base_dir: Path):
    from src.utils.logger import get_logger
    logger = get_logger("eda.pivot_tables", {"level": "INFO"})
    logger.info("=== Gerando Tabelas Dinâmicas de Risco — INÍCIO ===")

    # 1. Preparação
    df_raw = pd.read_parquet(base_dir / "data/processed/heart_disease.parquet")
    df = _add_binned_columns(df_raw, config, logger)
    target = config.get("target", "Heart Disease Status")
    df['target_numeric'] = df[target].map({'Yes': 1, 'No': 0})
    
    tables_dir = base_dir / "outputs/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 2. Pivot 01: Gênero x Faixa Etária -> Taxa de Risco
    t1 = df.pivot_table(index="Gender", columns="age_category", 
                        values="target_numeric", aggfunc="mean").round(4)
    t1.to_csv(tables_dir / "01_pivot_gender_age.csv")

    # 3. Pivot 02: Exercício x Tabagismo -> Taxa de Risco
    t2 = df.pivot_table(index="Exercise Habits", columns="Smoking", 
                        values="target_numeric", aggfunc="mean").round(4)
    t2.to_csv(tables_dir / "02_pivot_exercise_smoking.csv")

    # 4. Top 10 Pacientes com maior risco estatístico (Baseado em indicadores)
    # No seu caso, vamos pegar os 10 com maior Colesterol + Triglicerídeos
    top10 = df.nlargest(10, "Cholesterol Level").reset_index(drop=True)
    top10.to_csv(tables_dir / "06_top10_critical_patients.csv", index=False)

    logger.info(f"=== Tabelas Geradas com Sucesso em {tables_dir} ===")
    return {"tables_generated": 3}