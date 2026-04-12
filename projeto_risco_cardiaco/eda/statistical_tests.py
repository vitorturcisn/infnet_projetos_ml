import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Statsmodels para o teste de Tukey (Opcional)
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

# -- Bootstrap de Caminhos --
def _bootstrap_src_path(base_dir: Path) -> None:
    src_path = str(base_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

def run(config: dict, base_dir: Path):
    _bootstrap_src_path(base_dir)
    from src.utils.logger import get_logger
    logger = get_logger("eda.statistical_tests", {"level": "INFO"})

    logger.info("=== Iniciando Testes de Hipóteses Estatísticas ===")
    
    # 1. Carga
    df = pd.read_parquet(base_dir / "data/processed/heart_disease.parquet")
    target = config.get("target", "Heart Disease Status")
    num_cols = config.get("features", {}).get("numerical", [])
    out_dir = base_dir / "outputs/stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 2. Testes de Normalidade (Shapiro-Wilk)
    norm_records = []
    for col in num_cols:
        # Shapiro é sensível a amostras grandes, usamos 500 como no padrão do professor
        stat, p = scipy_stats.shapiro(df[col].dropna().sample(min(500, len(df))))
        norm_records.append({
            "feature": col, 
            "test": "shapiro", 
            "p_value": float(p), 
            "is_normal": bool(p > 0.05)
        })
    pd.DataFrame(norm_records).to_csv(out_dir / "09_normality_tests.csv", index=False)
    results["normality"] = norm_records

    # 3. ANOVA (Com a correção de salvamento que faltava)
    anova_results = []
    for col in num_cols:
        try:
            groups = [grp[col].dropna().values for name, grp in df.groupby(target)]
            f_stat, p_val = scipy_stats.f_oneway(*groups)
            anova_results.append({
                "feature": col, 
                "f_stat": float(f_stat), 
                "p_value": float(p_val), 
                "significant": bool(p_val < 0.05)
            })
        except Exception as e:
            logger.warning(f"Falha no ANOVA para {col}: {e}")
    
    # SALVANDO O RESULTADO DO ANOVA (Linha que faltava)
    with open(out_dir / "10_anova_results.json", "w") as f:
        json.dump(anova_results, f, indent=2)
    results["anova"] = anova_results

    # 4. Chi-Quadrado
    chi_records = []
    for pair in config.get("statistical_tests", {}).get("chi2_pairs", []):
        try:
            contingency = pd.crosstab(df[pair[0]], df[pair[1]])
            chi2, p, dof, ex = scipy_stats.chi2_contingency(contingency)
            chi_records.append({
                "pair": f"{pair[0]} x {pair[1]}", 
                "p_value": float(p), 
                "significant": bool(p < 0.05)
            })
        except Exception as e:
            logger.warning(f"Falha no Chi2 para {pair}: {e}")
            
    pd.DataFrame(chi_records).to_csv(out_dir / "16_chi2_tests.csv", index=False)
    results["chi_square"] = chi_records

    logger.info(f"=== Testes Concluídos! Resultados salvos em {out_dir} ===")
    return results