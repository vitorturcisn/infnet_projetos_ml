# %%
#
# Este script é a QUARTA etapa do pipeline de dados.
#
# Entrada : data/features/heart_disease_features.parquet  ← preprocessamento.py
# Saída   : mlruns.db  (tracking SQLite do MLFlow)
#           outputs/modeling/ (plots PNG e relatórios)
#
# TODA a política de experimentação (quais modelos, search_spaces, nº de
# trials, artefatos, CV) é definida em config/modeling.yaml.
# ─────────────────────────────────────────────────────────────────────────────

# %%
import sys
from pathlib import Path

# Garante que o sys.path ache a pasta raiz para os imports funcionarem
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
# Inicializa contexto do pipeline
from src.core.context import PipelineContext

context = PipelineContext.from_notebook(__file__)

# %%
# Importa e Executa a etapa completa de modelagem
try:
    from src.modeling.step import ModelingStep
    
    step = ModelingStep(context)
    step.run()
    
except ImportError as e:
    context.logger.error("Erro de Importação: %s", str(e))
    context.logger.warning("Verifique se os arquivos dentro da pasta 'src/modeling/' já foram criados e configurados corretamente.")