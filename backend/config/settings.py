"""Configuration settings for the automobile data analysis application."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application configuration settings."""
    
    # Mistral AI API Configuration
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_API_URL: str = os.getenv("MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-7b-instruct")
    
    # Chart Configuration
    DEFAULT_CHART_SIZE: tuple = (10, 6)
    DEFAULT_DPI: int = 300
    CHART_STYLE: str = "seaborn-v0_8"
    
    # Data Configuration
    DEFAULT_DATASET_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sales_data_sample.csv")
    MAX_CHART_POINTS: int = 1000
    
    # API Configuration
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    @classmethod
    def validate_settings(cls) -> Dict[str, Any]:
        """Validate required settings and return status."""
        validation_results = {
            "valid": True,
            "errors": []
        }
        
        if not cls.MISTRAL_API_KEY:
            validation_results["valid"] = False
            validation_results["errors"].append("MISTRAL_API_KEY is required")
            
        return validation_results

# Global settings instance
settings = Settings()
