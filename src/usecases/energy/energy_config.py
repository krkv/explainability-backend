"""Energy use case configuration."""

from pydantic import BaseModel
from pathlib import Path
from typing import List


class EnergyConfig(BaseModel):
    """Configuration for energy use case."""
    
    instance_path: Path = Path("instances/energy")
    model_path: Path = Path("instances/energy/model/custom_gp_model.pkl")
    dataset_path: Path = Path("instances/energy/data/summer_workday_test.csv")
    functions_json_path: Path = Path("instances/energy/functions.json")
    
    # DiCE configuration
    dice_continuous_features: List[str] = ['indoor_temperature', 'outdoor_temperature', 'past_electricity']
    dice_categorical_features: List[str] = []
    dice_outcome_name: str = 'prediction'
    dice_model_type: str = 'regressor'
    
    # SHAP configuration
    shap_explainer_type: str = 'kernel'
    shap_link: str = 'identity'
    shap_sample_size: int = 25

