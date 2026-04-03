from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Detector"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    app_port: int = 8002
    database_url: str = "sqlite+aiosqlite:///./ai_detector.db"

    # Claude API — deixe vazio para desabilitar o fallback para Claude
    anthropic_api_key: str = ""

    # Zona de incerteza do ML que ativa o fallback para Claude (texto)
    claude_confidence_threshold: float = 0.15  # |score - 0.5| < threshold → chama Claude

    # Quantos exemplos novos acumulam antes de disparar retrain do Random Forest
    retrain_threshold: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Singleton: mesma instância retornada em todas as chamadas (sem re-ler .env)
@lru_cache
def get_settings() -> Settings:
    return Settings()
