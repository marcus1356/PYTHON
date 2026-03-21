from fastapi import APIRouter, HTTPException, Query

from app.core.exceptions import (
    SubmissionAlreadyAnalyzingError,
    SubmissionNotFoundError,
)
from app.dependencies import DbSession
from app.schemas.common import PaginatedResponse
from app.schemas.submission import (
    SubmissionCreate,
    SubmissionDetail,
    SubmissionSummary,
    SubmissionUpdate,
)
from app.services.submission_service import SubmissionService

router = APIRouter(prefix="/submissions", tags=["submissions"])


def _get_service(session: DbSession) -> SubmissionService:
    return SubmissionService(session)


@router.post("", response_model=SubmissionSummary, status_code=201)
async def create_submission(body: SubmissionCreate, session: DbSession):
    """Cria uma nova submissão de texto para análise futura."""
    service = _get_service(session)
    return await service.create(body)


@router.get("", response_model=PaginatedResponse[SubmissionSummary])
async def list_submissions(
    session: DbSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: str | None = Query(default=None),
    source: str | None = Query(default=None),
):
    """Lista submissões com paginação e filtros opcionais."""
    service = _get_service(session)
    return await service.list_paginated(page, page_size, status, source)


@router.get("/{submission_id}", response_model=SubmissionDetail)
async def get_submission(submission_id: str, session: DbSession):
    """Retorna detalhes completos de uma submissão, incluindo resultado da análise."""
    service = _get_service(session)
    try:
        return await service.get_detail(submission_id)
    except SubmissionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{submission_id}", response_model=SubmissionSummary)
async def update_submission(
    submission_id: str,
    body: SubmissionUpdate,
    session: DbSession,
):
    """Atualiza campos mutáveis (title, source). text_content é imutável."""
    service = _get_service(session)
    try:
        return await service.update(submission_id, body)
    except SubmissionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{submission_id}", status_code=204)
async def delete_submission(submission_id: str, session: DbSession):
    """Deleta uma submissão. Não é possível deletar enquanto status=analyzing."""
    service = _get_service(session)
    try:
        await service.delete(submission_id)
    except SubmissionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SubmissionAlreadyAnalyzingError as e:
        raise HTTPException(status_code=409, detail=str(e))
