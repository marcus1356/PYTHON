from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

# Atalho de tipo para injeção de sessão nos endpoints
DbSession = Annotated[AsyncSession, Depends(get_db)]
