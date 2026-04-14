from sklearn.pipeline import Pipeline
from src.preprocessing.transformers import (
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    PolynomialFeatureTransformer,
    CardiacCategoricalEncoder,
    FeatureSelector
)

class PreprocessingPipelineBuilder:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger

    def build(self) -> Pipeline:
        # A ordem segue as dependências: Flags e Ratios primeiro, 
        # depois Polinômios que podem usar esses valores, e Seleção por último.
        steps = [
            ("flags", BinaryFlagTransformer(self.config.get("binary_flags", []), self.logger)),
            ("ratios", RatioFeatureTransformer(self.config.get("ratio_features", []), self.logger)),
            ("log", LogTransformer(self.config.get("log_transform", {}).get("columns", []), self.logger)),
            ("polinomiais", PolynomialFeatureTransformer(self.config.get("polynomial_features", []), self.logger)), # <-- Adicionado
            ("encoding", CardiacCategoricalEncoder(self.config.get("categorical_encoding", {}), self.logger)),
            ("selecao", FeatureSelector(self.config.get("feature_selection", {}).get("features_to_keep", []), self.logger))
        ]
        
        if self.logger:
            self.logger.info(f"Pipeline montado com {len(steps)} etapas.")
            
        return Pipeline(steps)