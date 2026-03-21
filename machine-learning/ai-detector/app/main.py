from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.database import Base, engine

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan é o padrão moderno do FastAPI (substituiu @app.on_event).
    Código antes do yield → startup.
    Código depois do yield → shutdown.
    """
    # Cria as tabelas no banco (apenas em desenvolvimento)
    # Em produção, use: alembic upgrade head
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    description="Plataforma para detectar conteúdo gerado por Inteligência Artificial.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/health", tags=["health"])
async def health_check():
    """Endpoint de saúde da aplicação."""
    return {"status": "ok", "version": "1.0.0", "port": settings.app_port}
