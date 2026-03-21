from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Repositório genérico com operações CRUD básicas.

    Ensino: Python Generics (Generic[ModelType]) permitem que essa classe
    sirva qualquer model ORM sem duplicar código. Semelhante aos generics
    de Java/C#, mas com tipagem verificada pelo mypy.
    """

    def __init__(self, model: type[ModelType], session: AsyncSession) -> None:
        self.model = model
        self.session = session

    async def get_by_id(self, id: str) -> ModelType | None:
        result = await self.session.get(self.model, id)
        return result

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 20,
        filters: list[Any] | None = None,
    ) -> tuple[list[ModelType], int]:
        stmt = select(self.model)
        count_stmt = select(func.count()).select_from(self.model)

        if filters:
            for f in filters:
                stmt = stmt.where(f)
                count_stmt = count_stmt.where(f)

        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = stmt.offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return items, total

    async def create(self, data: dict[str, Any]) -> ModelType:
        instance = self.model(**data)
        self.session.add(instance)
        await self.session.flush()  # obtém o ID sem fechar a sessão
        await self.session.refresh(instance)
        return instance

    async def update(self, instance: ModelType, data: dict[str, Any]) -> ModelType:
        for field, value in data.items():
            setattr(instance, field, value)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def delete(self, instance: ModelType) -> None:
        await self.session.delete(instance)
        await self.session.flush()
