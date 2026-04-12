import yaml
from pathlib import Path
from typing import Any

def load_yaml(path: Path) -> dict[str, Any]:
    """Carrega um arquivo YAML específico dado um Path completo."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Carrega e mescla as configurações principais (data + pipeline).
    Utilizado pelo orquestrador de EDA e pelo PipelineContext.
    """
    base_path = Path(config_path)
    if base_path.is_dir():
        # Se for um diretório, tenta carregar os arquivos padrão
        data_cfg = load_yaml(base_path / "data.yaml")
        pipe_cfg = load_yaml(base_path / "pipeline.yaml")
        # Mescla os dicionários
        return {**data_cfg, **pipe_cfg}
    else:
        # Se for um caminho para um arquivo específico, carrega ele
        return load_yaml(base_path)