"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
from typing import Optional


GOOGLE_PROJECT_ID = "explainability-assistant"
GOOGLE_LOCATION = "global"


class Model(str, Enum):
    """Supported LLM models."""
    GEMINI_3_1_FLASH_LITE_PREVIEW = "gemini-3.1-flash-lite-preview"
    GPT_5_4_MINI = "gpt-5.4-mini"


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

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )
    
    # LLM Configuration
    hf_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Paths
    instances_path: str = "instances"
    energy_instance_path: str = "instances/energy"
    heart_instance_path: str = "instances/heart"
    
    # Model Paths
    energy_model_path: str = "instances/energy/model/custom_gp_model.pkl"
    heart_model_path: str = "instances/heart/model/heart_model.pkl"
    
    # Dataset Paths
    energy_dataset_path: str = "instances/energy/data/summer_workday_test.csv"
    heart_dataset_path: str = "instances/heart/data/test_set.csv"
    
    # Logging
    log_level: str = "INFO"

    # Langfuse
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_base_url: Optional[str] = None
    langfuse_host: Optional[str] = None
    langfuse_tracing_environment: Optional[str] = None
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_title: str = "Explainability Assistant Backend"
    api_description: str = "LLM-powered assistant for ML model explanations"
    api_version: str = "2.0.0"


# Global settings instance
settings = Settings()
