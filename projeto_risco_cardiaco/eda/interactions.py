import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -- Bootstrap de Caminhos --
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE / "src") not in sys.path:
    sys.path.insert(0, str(_BASE / "src"))

def run(config: dict, base_dir: Path):
    from utils.logger import get_logger
    logger = get_logger("eda.interactions", {"level": "INFO"})
    logger.info("=== Analisando Interações de Risco — INÍCIO ===")

    # 1. Preparação
    df = pd.read_parquet(base_dir / "data/processed/heart_disease.parquet")
    target = config.get("target", "Heart Disease Status")
    df['target_numeric'] = df[target].map({'Yes': 1, 'No': 0})
    
    n_bins = config.get("interactions", {}).get("interaction_bins", 4)
    out_figures = base_dir / "outputs/figures"
    out_stats = base_dir / "outputs/stats"

    # 2. Criar categorias (bins) para variáveis numéricas
    # Isso transforma Idade em "Faixas Etárias" para o gráfico ficar legível
    for col in ["Age", "BMI", "Cholesterol Level"]:
        df[f"{col}_bin"] = pd.qcut(df[col], q=n_bins, duplicates='drop').astype(str)

    # 3. FIGURA 16: Interação Idade x IMC (Heatmap de Prevalência)
    plt.figure(figsize=(10, 8))
    pivot = df.pivot_table(index="Age_bin", columns="BMI_bin", 
                           values="target_numeric", aggfunc="mean")
    sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".2%")
    plt.title("Interação Idade x IMC: % de Risco Cardíaco")
    plt.savefig(out_figures / "fig_16_interaction_age_bmi.png")
    plt.close()

    # 4. FIGURA 21: Interação 3-Way (Idade x Fumante x Gênero)
    # Mostra como o tabagismo afeta diferentes idades e gêneros
    g = sns.FacetGrid(df, col="Gender", hue="Smoking", height=5, aspect=1.2)
    g.map_dataframe(sns.lineplot, x="Age_bin", y="target_numeric", marker="o")
    g.add_legend(title="Fumante?")
    g.set_axis_labels("Faixa Etária", "Taxa de Doença (%)")
    plt.savefig(out_figures / "fig_21_3way_age_smoking_gender.png")
    plt.close()

    # 5. Exportar Estatísticas (Igual ao 18_interaction_2way_means do professor)
    stats_2way = df.groupby(["Age_bin", "BMI_bin"])["target_numeric"].mean().reset_index()
    stats_2way.to_csv(out_stats / "18_interaction_2way_means.csv", index=False)

    logger.info("=== Análise de Interações concluída! ===")
    return {"stats_path": str(out_stats / "18_interaction_2way_means.csv")}