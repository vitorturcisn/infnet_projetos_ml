# %%
#
# SEGUNDA etapa do pipeline.
# Entrada : data/processed/heart_disease.parquet  ← gerado por ingestao.py
# Saída   : outputs/quality/quality_report_<timestamp>.json
#
# Conceito central: SEPARAÇÃO entre política e mecanismo
#   • Política  → config/quality.yaml      (O QUÊ validar e com quais thresholds)
#   • Mecanismo → src/quality/             (COMO executar as validações via GE)
#
# Para ajustar qualquer critério de qualidade (ex: range de Idade, tolerância a nulos), 
# edite apenas o YAML. Este arquivo não precisa mudar.
# ─────────────────────────────────────────────────────────────────────────────

# %%
import sys
from pathlib import Path

# Bootstrap: Garante que o diretório raiz esteja no sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
from src.core.context import PipelineContext
from src.quality.step import QualityStep

# Inicializa o contexto do pipeline
context = PipelineContext.from_notebook(__file__)

# Executa a etapa de qualidade
# Nota: Usamos a instanciação explícita para garantir que funcione sem 
# depender da configuração interna do run_step do professor.
step = QualityStep(context)
step.run()