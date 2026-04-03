"""Repositório para TrainingExample — buffer de aprendizado contínuo."""

import json
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.training_example import TrainingExample


class TrainingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, data: dict) -> TrainingExample:
        obj = TrainingExample(**data)
        self.session.add(obj)
        await self.session.commit()
        await self.session.refresh(obj)
        return obj

    async def count_unused(self) -> int:
        result = await self.session.execute(
            select(func.count()).where(TrainingExample.used_in_training == False)  # noqa: E712
        )
        return result.scalar_one()

    async def get_all_for_training(self) -> list[dict]:
        """Retorna todos os exemplos como lista de dicts (features_json + label)."""
        result = await self.session.execute(
            select(TrainingExample.features_json, TrainingExample.label)
        )
        return [{"features_json": row.features_json, "label": row.label} for row in result]

    async def mark_all_as_used(self) -> None:
        from sqlalchemy import update
        await self.session.execute(
            update(TrainingExample).values(used_in_training=True)
        )
        await self.session.commit()

    async def get_by_detection_id(self, detection_id: str) -> TrainingExample | None:
        result = await self.session.execute(
            select(TrainingExample).where(TrainingExample.detection_id == detection_id)
        )
        return result.scalar_one_or_none()
