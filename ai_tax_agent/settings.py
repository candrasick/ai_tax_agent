"""
Configuration settings for the AI Tax Agent, loaded from environment variables and .env files.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings.
    
    Attributes:
        gemini_api_key (str): The API key for accessing Google Gemini models.
    """
    gemini_api_key: str
    database_url: str = "sqlite:///tax_data.db" # Default database URL

    # Configure Pydantic to load from a .env file
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

# Create a single instance of the settings to be imported across the application
settings = Settings() 