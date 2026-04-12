import sys
from pathlib import Path
import pandas as pd

# ── AJUSTE DE CAMINHO ──
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent # Sobe de 'utils' para 'production_app' e depois para a raiz

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── IMPORTS MODULARES (VERIFIQUE A GRAFIA) ──
from src.preprocessing.transformers.log_transform import LogTransformer
from src.preprocessing.transformers.ratio_features import RatioFeatureTransformer
from src.preprocessing.transformers.polynomial_features import PolynomialFeatureTransformer
from src.preprocessing.transformers.categorical_encoder import CardiacCategoricalEncoder

# Caminhos dos dados
_DATA_DIR = _PROJECT_ROOT / "data"
_PARQUET_FEATURES = _DATA_DIR / "features" / "heart_disease_features.parquet"
_TARGET_COL = "target_numeric"

# No production_app/utils/pipeline_utils.py

def preprocessar_entradas(raw_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_dict])
    
    # 1. ENCODING
    df = CardiacCategoricalEncoder(
        enc_config={
            "columns": ["Stress Level", "Gender", "Diabetes", "Smoking"],
            "one_hot_prefix": "cat"
        }
    ).transform(df)

    # 2. FLAGS BINÁRIAS
    df["is_hypertensive"] = 0 
    df["is_obese"] = (df["BMI"] > 30).astype(int)

    # 3. RAZÕES, LOG E POLINOMIAIS
    df = RatioFeatureTransformer(ratios=[
        {"name": "cholesterol_bmi_ratio", "numerator": "Cholesterol Level", "denominator": "BMI"},
        {"name": "triglyceride_age_index", "numerator": "Triglyceride Level", "denominator": "Age"}
    ]).transform(df)
    
    df = LogTransformer(columns=["Triglyceride Level", "CRP Level", "Homocysteine Level"]).transform(df)
    
    df = PolynomialFeatureTransformer(poly_config=[
        {"name": "age_x_bmi", "columns": ["Age", "BMI"]}
    ]).transform(df)

    # 4. 🚀 A ORDEM SAGRADA (Copiada exatamente do seu Debug)
    features_obrigatorias = [
        'Age', 'BMI', 'is_obese', 'is_hypertensive', 
        'cholesterol_bmi_ratio', 'triglyceride_age_index', 'age_x_bmi', 
        'log_Triglyceride Level', 'Stress Level_encoded', 
        'cat_Gender_Male', 'cat_Gender_Female', 
        'cat_Smoking_Yes', 'cat_Diabetes_Yes'
    ]

    # O reindex garante que a ordem seja IDENTICA à do treino
    df_final = df.reindex(columns=features_obrigatorias, fill_value=0)
    
    return df_final

def obter_parquet_features() -> pd.DataFrame:
    return pd.read_parquet(_PARQUET_FEATURES)

# No final do seu production_app/utils/pipeline_utils.py

def obter_colunas_features_brutas() -> list[str]:
    return [
        "Age", "BMI", "log_Triglyceride Level", "cholesterol_bmi_ratio", 
        "triglyceride_age_index", "age_x_bmi", "Stress Level_encoded",
        "cat_Diabetes_Yes", "cat_Gender_Female", "cat_Gender_Male", 
        "cat_Smoking_Yes", "is_hypertensive", "is_obese"  # <── Adicione aqui!
    ]