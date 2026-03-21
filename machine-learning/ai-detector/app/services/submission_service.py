import math

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    SubmissionAlreadyAnalyzedError,
    SubmissionAlreadyAnalyzingError,
    SubmissionNotFoundError,
)
from app.models.submission import Submission
from app.repositories.analysis_repo import AnalysisRepository
from app.repositories.submission_repo import SubmissionRepository
from app.schemas.common import PaginatedResponse
from app.schemas.submission import SubmissionCreate, SubmissionDetail, SubmissionSummary, SubmissionUpdate
from app.services import detection_service


class SubmissionService:
    """
    Camada de regras de negócio para submissões.
    Conhece: repositórios, regras de estado, lógica de detecção.
    NÃO conhece: HTTP, status codes, FastAPI.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.repo = SubmissionRepository(session)
        self.analysis_repo = AnalysisRepository(session)

    async def create(self, data: SubmissionCreate) -> SubmissionSummary:
        words = data.text_content.split()
        db_obj = await self.repo.create({
            "text_content": data.text_content,
            "title": data.title,
            "source": data.source,
            "word_count": len(words),
            "char_count": len(data.text_content),
            "status": "pending",
        })
        return SubmissionSummary.model_validate(db_obj)

    async def list_paginated(
        self,
        page: int,
        page_size: int,
        status: str | None,
        source: str | None,
    ) -> PaginatedResponse[SubmissionSummary]:
        items, total = await self.repo.get_all_paginated(page, page_size, status, source)
        pages = math.ceil(total / page_size) if page_size else 1
        return PaginatedResponse(
            items=[SubmissionSummary.model_validate(s) for s in items],
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )

    async def get_detail(self, submission_id: str) -> SubmissionDetail:
        obj = await self.repo.get_by_id(submission_id)
        if not obj:
            raise SubmissionNotFoundError(submission_id)
        return SubmissionDetail.model_validate(obj)

    async def update(self, submission_id: str, data: SubmissionUpdate) -> SubmissionSummary:
        obj = await self.repo.get_by_id(submission_id)
        if not obj:
            raise SubmissionNotFoundError(submission_id)

        updates = data.model_dump(exclude_none=True)
        updated = await self.repo.update(obj, updates)
        return SubmissionSummary.model_validate(updated)

    async def delete(self, submission_id: str) -> None:
        obj = await self.repo.get_by_id(submission_id)
        if not obj:
            raise SubmissionNotFoundError(submission_id)
        if obj.status == "analyzing":
            raise SubmissionAlreadyAnalyzingError(submission_id)
        await self.repo.delete(obj)

    async def analyze(self, submission_id: str, force: bool = False) -> Submission:
        obj = await self.repo.get_by_id(submission_id)
        if not obj:
            raise SubmissionNotFoundError(submission_id)
        if obj.status == "analyzing":
            raise SubmissionAlreadyAnalyzingError(submission_id)
        if obj.status == "completed" and not force:
            raise SubmissionAlreadyAnalyzedError(submission_id)

        # Se force=True, remove resultado anterior para não violar UNIQUE constraint
        if force:
            existing = await self.analysis_repo.get_by_submission_id(submission_id)
            if existing:
                await self.analysis_repo.delete(existing)
                # Expira o cache de relações do objeto para evitar referência ao resultado deletado
                self.repo.session.expire(obj)

        # Marca como analisando
        await self.repo.update(obj, {"status": "analyzing"})

        try:
            result_data = detection_service.analyze_text(obj.text_content)
            result_data["submission_id"] = submission_id

            await self.analysis_repo.create(result_data)
            await self.repo.update(obj, {"status": "completed"})
        except Exception:
            await self.repo.update(obj, {"status": "failed"})
            raise

        return obj
