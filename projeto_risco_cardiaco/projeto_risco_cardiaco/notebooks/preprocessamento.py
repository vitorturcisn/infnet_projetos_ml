# %%
#
# TERCEIRA etapa do pipeline de dados.
#   Entrada : data/processed/heart_disease.parquet  ← gerado por qualidade.py
#   Saída   : data/features/heart_disease_features.parquet
#
# Conceito central: SEPARAÇÃO entre política e mecanismo
#   • Política  → config/preprocessing.yaml  (O QUÊ transformar e parâmetros)
#   • Mecanismo → src/preprocessing/         (COMO executar cada transformação)
#
# Transformações orchestradas por PreprocessingStep (na ordem correta):
#   1. Flags binárias      — is_hypertensive, is_obese (seguras para inferência)
#   2. Features de razão   — cholesterol_bmi_ratio, triglyceride_age_index
#   3. Transformação log1p — reduz assimetria em Triglicérides e CRP
#   4. Features de interação — age_x_bmi (captura o risco detectado no EDA)
#   5. Encoding categórico — One-hot e Ordinal para Stress, Gender, etc.
#   6. Seleção de features — subconjunto final definido no YAML
#
# ⚠  Data Leakage — transformadores stateful (StandardScalerTransformer) 
#    NÃO são aplicados aqui. Eles ficam no pipeline de modelagem, após o split.
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Configura o contexto de execução (caminhos, config, logger)
import sys
from pathlib import Path

# Bootstrap: garante que root_dir esteja no sys.path antes de qualquer import de src/
# Como este arquivo está em 'notebooks/', o parent.parent volta para a raiz do projeto
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.core.context import PipelineContext
from src.preprocessing import PreprocessingStep

# PipelineContext.from_notebook resolve a raiz do projeto a partir do __file__
# e garante que src/ e config/ estejam no sys.path.
ctx = PipelineContext.from_notebook(__file__)

# %%
# Executa a etapa completa:
#   1. Carrega data/processed/heart_disease.parquet
#   2. Constrói o sklearn.Pipeline a partir de config/preprocessing.yaml
#   3. Aplica fit_transform (todas as etapas stateless)
#   4. Persiste o resultado em data/features/heart_disease_features.parquet
#   5. Loga schema, shape, valores ausentes e métricas de saída
step = PreprocessingStep(ctx)
step.run()