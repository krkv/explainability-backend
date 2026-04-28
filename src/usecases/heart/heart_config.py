"""Heart use case configuration."""

from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional


class HeartConfig(BaseModel):
    """Configuration for heart use case."""
    
    instance_path: Path = Path("instances/heart")
    model_path: Path = Path("instances/heart/model/heart_model.pkl")
    model_metadata_path: Path = Path("instances/heart/model/heart_model_metadata.json")
    dataset_path: Path = Path("instances/heart/data/test_set.csv")
    dataset_metadata_path: Path = Path("instances/heart/data/dataset_metadata.json")
    feature_metadata_path: Path = Path("instances/heart/data/feature_metadata.json")
    functions_json_path: Path = Path("instances/heart/functions.json")
    shap_cache_path: Path = Path("instances/heart/data/shap_cache.pkl")
    cf_cache_path: Path = Path("instances/heart/data/cf_cache.pkl")
    global_fi_cache_path: Path = Path("instances/heart/data/global_fi_cache.pkl")
    
    # Dataset configuration
    target_variable: str = 'num'
    class_names: List[str] = ["NEGATIVE", "POSITIVE"]
    
    # DiCE configuration
    dice_continuous_features: List[str] = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    dice_categorical_features: List[str] = ["thal", "ca", "slope", "restecg", "cp", "sex", "fbs", "exang"]
    dice_outcome_name: str = 'prediction'
    dice_model_type: str = 'classifier'
    
    # SHAP configuration
    shap_explainer_type: str = 'kernel'
    shap_sample_size: int = 25
