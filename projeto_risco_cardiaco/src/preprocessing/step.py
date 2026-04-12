import pandas as pd
from src.core.base import PipelineStep
from src.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from src.utils.config_loader import load_yaml

class PreprocessingStep(PipelineStep):
    def __init__(self, context):
        super().__init__(logger=context.logger)
        self.context = context
        self.config = load_yaml(context.config_dir / "preprocessing.yaml")

    def run(self):
        self.logger.info("=== Iniciando Step de Pré-processamento Cardíaco ===")
        
        # 1. Carga
        df = pd.read_parquet(self.context.output_path) # Caminho vindo do step anterior
        df['target_numeric'] = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})

        # 2. Transformação
        builder = PreprocessingPipelineBuilder(self.config, self.logger)
        pipeline = builder.build()
        df_out = pipeline.fit_transform(df)

        # 3. Persistência
        out_path = self.context.root_dir / "data/features/heart_disease_features.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(out_path, compression="snappy")
        
        self.logger.info("✅ Pré-processamento concluído. Shape final: %s", df_out.shape)