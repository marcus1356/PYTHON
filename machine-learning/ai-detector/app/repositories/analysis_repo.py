from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis_result import AnalysisResult
from app.repositories.base import BaseRepository


class AnalysisRepository(BaseRepository[AnalysisResult]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(AnalysisResult, session)

    async def get_by_submission_id(self, submission_id: str) -> AnalysisResult | None:
        stmt = select(AnalysisResult).where(
            AnalysisResult.submission_id == submission_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
