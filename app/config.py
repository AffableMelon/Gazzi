import os
from pathlib import Path
from dotenv import load_dotenv

# Find workspace root
BASE_DIR = Path(__file__).resolve().parents[1]

# Load .env file
load_dotenv(BASE_DIR / ".env")

class Settings:
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    SESSIONS_DIR: Path = BASE_DIR / "data" / "hil_sessions"
    EXPORTS_DIR: Path = BASE_DIR / "data" / "exports"
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    PRECOMPUTED_DIR: Path = BASE_DIR / "data" / "precomputed"
    STATIC_DIR: Path = BASE_DIR / "app" / "frontend"

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Execution Settings
    FALLBACK_TO_LOCAL_OCR: bool = os.getenv("FALLBACK_TO_LOCAL_OCR", "True").lower() == "true"
    MAX_PAGES_PER_BATCH: int = int(os.getenv("MAX_PAGES_PER_BATCH", "5"))
    SCORE_THRESHOLD: int = int(os.getenv("SCORE_THRESHOLD", "95"))

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    def ensure_directories(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
