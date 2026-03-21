from fastapi import APIRouter, HTTPException, Query

from app.core.exceptions import (
    AnalysisResultNotFoundError,
    SubmissionAlreadyAnalyzedError,
    SubmissionAlreadyAnalyzingError,
    SubmissionNotFoundError,
)
from app.dependencies import DbSession
from app.repositories.analysis_repo import AnalysisRepository
from app.schemas.analysis_result import AnalysisResultResponse
from app.schemas.common import AnalyzeResponse
from app.services.submission_service import SubmissionService

router = APIRouter(prefix="/submissions", tags=["analysis"])


@router.post("/{submission_id}/analyze", response_model=AnalyzeResponse, status_code=202)
async def analyze_submission(
    submission_id: str,
    session: DbSession,
    force_reanalyze: bool = Query(default=False),
):
    """
    Dispara a análise de detecção de IA para uma submissão.

    Retorna 202 Accepted: preparado para Fase 2 onde a análise será assíncrona.
    Em Fase 1, a análise ocorre de forma síncrona mas o contrato HTTP já está correto.
    """
    service = SubmissionService(session)
    try:
        await service.analyze(submission_id, force=force_reanalyze)
        return AnalyzeResponse(
            submission_id=submission_id,
            status="completed",
            message="Analysis completed successfully.",
        )
    except SubmissionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SubmissionAlreadyAnalyzingError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except SubmissionAlreadyAnalyzedError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/{submission_id}/result", response_model=AnalysisResultResponse)
async def get_analysis_result(submission_id: str, session: DbSession):
    """Retorna o resultado da análise de IA para uma submissão."""
    analysis_repo = AnalysisRepository(session)
    result = await analysis_repo.get_by_submission_id(submission_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis result for submission '{submission_id}'. Submit to /analyze first.",
        )
    return AnalysisResultResponse.from_orm_model(result)
