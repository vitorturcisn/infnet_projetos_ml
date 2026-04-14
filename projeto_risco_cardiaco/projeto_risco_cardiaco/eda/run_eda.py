import argparse
import sys
import time
from pathlib import Path
from typing import Any

# -- Bootstrap de Caminhos --
_THIS_FILE = Path(__file__).resolve()
_EDA_DIR   = _THIS_FILE.parent
_BASE_DIR  = _EDA_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# Registro de todas as etapas (Seguindo a ordem lógica do professor)
_STEPS = [
    ("descriptive", "eda.descriptive"),
    ("visualizations", "eda.visualizations"),
    ("pivot_tables", "eda.pivot_tables"),
    ("statistical_tests", "eda.statistical_tests"),
    ("interactions", "eda.interactions"),
    ("feature_engineering", "eda.feature_engineering"),
    ("clustering", "eda.clustering"),
]

def run_eda():
    pipeline_start = time.time()
    logger = get_logger("eda.orchestrator", {"level": "INFO"})
    
    logger.info("=" * 70)
    logger.info("🚀 INICIANDO PIPELINE DE EDA COMPLETO - RISCO CARDÍACO")
    logger.info("=" * 70)

    # Carrega a configuração unificada
    config = load_config("config") 
    
    # IMPORTANTE: Carrega o eda.yaml manualmente e mescla, pois o 
    # load_config padrão as vezes foca apenas em data/pipeline
    import yaml
    with open(_BASE_DIR / "config/eda.yaml", "r", encoding="utf-8") as f:
        eda_cfg = yaml.safe_load(f)
        config.update(eda_cfg)

    for step_name, module_path in _STEPS:
        logger.info(f">> Executando etapa: {step_name}")
        step_start = time.time()
        
        try:
            import importlib
            module = importlib.import_module(module_path)
            module.run(config, _BASE_DIR)
            logger.info(f"   [OK] {step_name} concluído em {time.time() - step_start:.1f}s")
        except Exception as e:
            logger.error(f"   [ERRO] Falha na etapa {step_name}: {e}")

    logger.info("=" * 70)
    logger.info(f"✨ EDA FINALIZADO COM SUCESSO EM {time.time() - pipeline_start:.1f}s")
    logger.info("=" * 70)

if __name__ == "__main__":
    run_eda()