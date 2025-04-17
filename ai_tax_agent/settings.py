"""
Configuration settings for the AI Tax Agent, loaded from environment variables and .env files.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings.
    
    Attributes:
        gemini_api_key (str): The API key for accessing Google Gemini models.
        serp_api_key (str): The API key for accessing SerpAPI.
    """
    gemini_api_key: str | None = None
    database_url: str = "sqlite:///tax_data.db" # Default database URL
    serp_api_key: str | None = None # Added SerpAPI key

    # Configure Pydantic to load from a .env file
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

# Create a single instance of the settings to be imported across the application
settings = Settings() 