import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # FastAPI settings
    APP_NAME = "Financial Agents System"
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # LLM settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","ollama")
    # LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
    # OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")


    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./financial_agents.db")
    
    # Data settings
    DEFAULT_PERIOD = "7d"
    DEFAULT_INTERVAL = "1d"
    
    # Backtesting settings
    DEFAULT_START_DATE = "2022-01-01"
    DEFAULT_END_DATE = "2023-12-31"
    DEFAULT_INITIAL_CAPITAL = 100000
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "financial_agents.log")

config = Config()
