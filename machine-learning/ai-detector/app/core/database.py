from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# Engine assíncrono — echo=True mostra SQL no console em modo debug
engine = create_async_engine(settings.database_url, echo=settings.debug)

# expire_on_commit=False é CRÍTICO em async:
# sem isso, acessar atributos após commit dispara lazy load que falha em async
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency do FastAPI: fornece uma sessão de banco de dados por request.
    Commit automático ao final; rollback em caso de exceção.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
