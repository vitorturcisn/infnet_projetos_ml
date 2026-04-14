import json
import sys
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- Bootstrap de Caminhos ---
def _bootstrap_src_path(base_dir: Path) -> None:
    src_path = str(base_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

# --- Análise de Agrupamento de Pacientes ---
def _patient_clustering(df, config, random_state, dirs, logger):
    logger.info("Iniciando Agrupamento de Pacientes (K-Means)...")
    features = config.get("features", {}).get("numerical", [])
    
    # Identifica quem não tem valores vazios para não quebrar o algoritmo
    df_clean = df[features].dropna()
    X_scaled = StandardScaler().fit_transform(df_clean)
    
    # Testando o "Cotovelo" (Elbow Method) para achar o k ideal
    scores = []
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        # Usamos uma amostra para o cálculo do silhouette ser mais rápido
        sil = silhouette_score(X_scaled, labels, sample_size=min(1000, len(X_scaled)))
        scores.append({"k": k, "silhouette": float(sil), "inertia": float(km.inertia_)})
        logger.info(f"   K-Means k={k} | Silhouette: {sil:.4f}")

    # Salva os scores para análise posterior
    pd.DataFrame(scores).to_csv(dirs["stats"] / "23_clustering_scores.csv", index=False)
    
    # Treina o modelo final (escolhemos k=3 para representar risco Baixo, Médio e Alto)
    km_final = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    labels_final = km_final.fit_predict(X_scaled)
    
    # Alinha os resultados com os índices originais (quem era NaN continua NaN)
    cluster_series = pd.Series(np.nan, index=df.index)
    cluster_series.loc[df_clean.index] = labels_final
    
    return cluster_series, df_clean

def run(config: dict, base_dir: Path):
    _bootstrap_src_path(base_dir)
    from src.utils.logger import get_logger
    logger = get_logger("eda.clustering", {"level": "INFO"})

    # Configuração de pastas
    dirs = {
        "stats": base_dir / "outputs/stats",
        "figures": base_dir / "outputs/figures"
    }
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    # Carga de dados
    data_path = base_dir / "data/processed/heart_disease.parquet"
    df = pd.read_parquet(data_path)
    
    # Execução do Clustering
    random_seed = 42
    df["cluster_perfil"], df_clean = _patient_clustering(df, config, random_seed, dirs, logger)
    
    # PCA (Redução para 2D usando apenas os dados sem NaN)
    logger.info("Gerando visualização PCA 2D...")
    features = config.get("features", {}).get("numerical", [])
    X_scaled_clean = StandardScaler().fit_transform(df_clean)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled_clean)
    
    # Plotagem
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        components[:, 0], 
        components[:, 1], 
        c=df_clean.index.map(df["cluster_perfil"]), # Mapeia as cores corretamente
        cmap='viridis', 
        alpha=0.6
    )
    plt.colorbar(scatter, label='Cluster (Perfil)')
    plt.title("Perfis de Pacientes (Visualização PCA 2D)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    
    plt.savefig(dirs["figures"] / "fig_32_pca_scatter.png")
    plt.close()
    
    logger.info("=== Módulo de Clustering concluído com sucesso! ===")
    return {"cluster_labels": df["cluster_perfil"]}