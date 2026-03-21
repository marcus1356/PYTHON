from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.database import Base, engine

_STATIC = Path(__file__).resolve().parent.parent / "static"
_SANDBOX_HTML = _STATIC / "index.html"
_CHANGELOG_HTML = _STATIC / "changelog.html"

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)

@app.get("/sandbox", include_in_schema=False)
async def sandbox():
    """Frontend sandbox para testar a API."""
    return FileResponse(_SANDBOX_HTML, media_type="text/html")


@app.get("/changelog", include_in_schema=False)
async def changelog():
    """Página de release notes."""
    return FileResponse(_CHANGELOG_HTML, media_type="text/html")


@app.get("/health", tags=["health"])
async def health_check():
    """Endpoint de saúde da aplicação."""
    return {"status": "ok", "version": "1.0.0", "port": settings.app_port}
