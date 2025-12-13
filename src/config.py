"""Configuration module for Travel Assistant"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    POLICY_DIR = DATA_DIR / "policies"
    VECTOR_STORE_DIR = DATA_DIR / "vector_store"

    # Google Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"

    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION = "travel_policies"

    # Mem0
    MEM0_API_KEY = os.getenv("MEM0_API_KEY", None)

    # LangFuse
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", None)
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", None)
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Application
    USER_SESSION_EXPIRY = 3600  # 1 hour in seconds
    MAX_HISTORY_ITEMS = 50

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.POLICY_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(exist_ok=True)

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set in .env file. "
                "Please add your Google API key to the .env file.\n"
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )


# Ensure directories exist
Config.ensure_directories()

# Validate configuration (will raise error if GOOGLE_API_KEY is missing)
try:
    Config.validate()
except ValueError as e:
    import sys  # noqa: F401 - used for error messaging

    print(f"\n❌ Configuration Error: {e}\n")
    if "GOOGLE_API_KEY" in str(e):
        print("Setup Instructions:")
        print("1. Open the .env file in the project root")
        print("2. Add your Google API key: GOOGLE_API_KEY=your-actual-key-here")
        print("3. Save the file and try again\n")
    # Don't exit during import, just warn
    print("⚠️  Warning: Application may not work without proper configuration\n")
