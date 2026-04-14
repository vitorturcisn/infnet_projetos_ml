# %% [markdown]
# # Pipeline de Ingestão - Projeto Risco Cardíaco
# Este script automatiza o download do Kaggle e a conversão para Parquet.
# Toda a configuração é lida de config/data.yaml e config/pipeline.yaml.

# %%
import sys
from pathlib import Path

# 1. RESOLUÇÃO DE CAMINHOS
# Como este arquivo está em 'notebooks/', precisamos subir um nível para achar a pasta 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
from src.core.context import PipelineContext

# 2. INICIALIZAÇÃO DO CONTEXTO
# O método from_notebook identifica automaticamente a raiz do projeto e carrega os YAMLs
context = PipelineContext.from_notebook(__file__)

# 3. EXECUÇÃO DA ETAPA
# O comando abaixo vai disparar o download e depois a conversão para Parquet
context.run_step("ingestion")