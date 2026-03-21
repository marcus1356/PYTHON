from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Detector"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    app_port: int = 8001
    database_url: str = "sqlite+aiosqlite:///./ai_detector.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Singleton: mesma instância retornada em todas as chamadas (sem re-ler .env)
@lru_cache
def get_settings() -> Settings:
    return Settings()
