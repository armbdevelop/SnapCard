from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "SnapCard"
    debug: bool = False

    # Database
    database_url: str = "sqlite+aiosqlite:///./snapcard.db"

    # File storage
    upload_dir: Path = Path("uploads")
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: set[str] = {"jpg", "jpeg", "png", "webp"}

    # ML
    device: str = "cpu"
    blip_model: str = "Salesforce/blip-image-captioning-large"
    blip_lora_path: str | None = None
    clip_model: str = "openai/clip-vit-base-patch32"
    text_model: str = "google/mt5-base"
    text_model_fallback: str = "ai-forever/rugpt3small_based_on_gpt2"
    model_cache_dir: str = "./model_cache"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_prefix": "SNAPCARD_", "env_file": ".env"}


settings = Settings()
