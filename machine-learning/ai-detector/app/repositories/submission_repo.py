from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.submission import Submission
from app.repositories.base import BaseRepository


class SubmissionRepository(BaseRepository[Submission]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Submission, session)

    async def get_all_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
        source: str | None = None,
    ) -> tuple[list[Submission], int]:
        filters = []
        if status:
            filters.append(Submission.status == status)
        if source:
            filters.append(Submission.source == source)

        stmt = select(Submission).order_by(desc(Submission.created_at))
        count_stmt = select(Submission)

        from sqlalchemy import func
        count_q = select(func.count()).select_from(Submission)

        if filters:
            for f in filters:
                stmt = stmt.where(f)
                count_q = count_q.where(f)

        total_result = await self.session.execute(count_q)
        total = total_result.scalar_one()

        skip = (page - 1) * page_size
        stmt = stmt.offset(skip).limit(page_size)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())

        return items, total
