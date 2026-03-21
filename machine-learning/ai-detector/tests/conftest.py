"""
conftest.py — Base de testes do projeto.

Ensino:
- O banco de testes usa SQLite em memória (`:memory:`) → zero I/O, rápido.
- A dependency `get_db` do FastAPI é sobrescrita (override) para usar o banco de teste.
- Cada teste recebe um banco limpo (scope="function") → testes isolados, sem efeitos colaterais.
- O `AsyncClient` do httpx simula requests HTTP sem subir um servidor real.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Texto longo o suficiente para passar na validação (min_length=50)
SAMPLE_TEXT = (
    "Artificial intelligence has profoundly transformed modern industries. "
    "Machine learning models now process vast amounts of data with remarkable accuracy. "
    "Natural language processing enables computers to understand human communication. "
    "These advancements create both opportunities and challenges for society. "
    "Researchers continue to explore the boundaries of what AI systems can achieve."
)


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Sessão de banco de dados em memória — limpa a cada teste."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession):
    """
    Cliente HTTP de teste com dependency override.
    O FastAPI usa o banco de teste em vez do banco real.
    """
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture
def submission_payload(sample_text: str) -> dict:
    return {
        "text_content": sample_text,
        "title": "Test Submission",
        "source": "pytest",
    }
