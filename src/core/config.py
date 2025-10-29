"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from enum import Enum
from typing import Optional
from pathlib import Path


class Model(str, Enum):
    """Supported LLM models."""
    LLAMA_3_3_70B = "Llama-3.3-70B-Instruct"
    GEMINI_2_0_FLASH = "Gemini-2.0-Flash"
    
    @classmethod
    def from_string(cls, value: str) -> "Model":
        """Map legacy string format to enum."""
        mapping = {
            "Llama 3.3 70B Instruct": cls.LLAMA_3_3_70B,
            "Gemini 2.0 Flash": cls.GEMINI_2_0_FLASH,
        }
        return mapping.get(value, cls.LLAMA_3_3_70B)


class UseCase(str, Enum):
    """Supported use cases."""
    ENERGY = "energy"
    HEART = "heart"
    
    @classmethod
    def from_string(cls, value: str) -> "UseCase":
        """Map legacy string format to enum."""
        mapping = {
            "Heart Disease": cls.HEART,
            "Energy Consumption": cls.ENERGY,
        }
        return mapping.get(value, cls(value.lower()))


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    hf_token: Optional[str] = None
    google_project: str = "explainability-app"
    google_location: str = "europe-north1"
    
    # Paths
    instances_path: str = "instances"
    energy_instance_path: str = "instances/energy"
    heart_instance_path: str = "instances/heart"
    
    # Model Paths
    energy_model_path: str = "instances/energy/model/custom_gp_model.pkl"
    heart_model_path: str = "instances/heart/model/best_model_3_DecisionTreeClassifier.pkl"
    
    # Dataset Paths
    energy_dataset_path: str = "instances/energy/data/summer_workday_test.csv"
    heart_dataset_path: str = "instances/heart/data/test_set.csv"
    
    # Logging
    log_level: str = "INFO"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_title: str = "XAI LLM Chat Backend"
    api_description: str = "LLM-powered assistant for ML model explanations"
    api_version: str = "2.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
