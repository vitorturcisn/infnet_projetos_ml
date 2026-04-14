import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.smoothers_lowess import lowess

# -- Bootstrap de Caminhos (Padrão do Professor) --
_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))

def run(config: dict, base_dir: Path):
    from utils.logger import get_logger
    logger = get_logger("eda.visualizations", {"level": "INFO"})
    logger.info("=== Gerando Galeria de Gráficos de Risco — INÍCIO ===")

    # 1. Carga e Setup
    df = pd.read_parquet(base_dir / "data/processed/heart_disease.parquet")
    target = config.get("target", "Heart Disease Status")
    num_cols = config["features"]["numerical"]
    out_dir = base_dir / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Converter alvo para numérico para tendências
    df['target_numeric'] = df[target].map({'Yes': 1, 'No': 0})

    # 2. FIG 01: Distribuição do Alvo (KDE de Probabilidade)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=target, palette="viridis")
    plt.title(f"Distribuição de Frequência: {target}")
    plt.savefig(out_dir / "fig_01_target_distribution.png")

    # 3. FIG 03: Grid de Histogramas (Todas as variáveis numéricas)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, col in enumerate(num_cols[:9]):
        ax = axes.flatten()[i]
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribuição: {col}")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_03_feature_distributions.png")

    # 4. FIG 05: Heatmap de Correlação
    plt.figure(figsize=(12, 10))
    corr = df[num_cols + ['target_numeric']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlação Clínica")
    plt.savefig(out_dir / "fig_05_correlation_heatmap.png")

    # 5. FIG 12: Idade vs Risco (Tendência LOWESS) - O "Estado da Arte"
    plt.figure(figsize=(10, 6))
    # Amostra para não travar o gráfico
    sample = df.sample(min(2000, len(df)))
    plt.scatter(sample["Age"], sample["target_numeric"], alpha=0.1, color="gray")
    
    # Linha de tendência suave (LOWESS)
    filtered = df.groupby("Age")["target_numeric"].mean().reset_index()
    res = lowess(filtered["target_numeric"], filtered["Age"], frac=0.3)
    plt.plot(res[:, 0], res[:, 1], color="red", linewidth=3, label="Tendência de Risco")
    
    plt.title("Evolução do Risco por Idade (Suavização LOWESS)")
    plt.xlabel("Idade")
    plt.ylabel("Probabilidade de Doença")
    plt.legend()
    plt.savefig(out_dir / "fig_12_age_vs_risk_trend.png")

    logger.info(f"=== Galeria gerada! Verifique a pasta: {out_dir} ===")
    return [str(f) for f in out_dir.glob("*.png")]