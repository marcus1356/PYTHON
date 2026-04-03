"""
Endpoint unificado de detecção de IA.

POST /api/v1/detect    — analisa texto ou imagem
POST /api/v1/feedback  — usuário corrige o resultado → modelo aprende
"""

import json
import logging
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config import get_settings
from app.dependencies import DbSession
from app.repositories.training_repo import TrainingRepository
from app.schemas.detect import DetectRequest, DetectResponse, FeedbackRequest, FeedbackResponse
from app.services import claude_service, detection_service, learning_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detect", tags=["detect"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blend_with_claude(base_score: float, claude_prob: float) -> float:
    """Blenda o score ML com o resultado do Claude: 40% ML + 60% Claude."""
    return round(0.40 * base_score + 0.60 * claude_prob, 4)


def _score_to_verdict(score: float) -> str:
    if score >= 0.60:
        return "ai"
    if score <= 0.40:
        return "human"
    return "uncertain"


def _score_to_confidence(score: float) -> str:
    if score < 0.30 or score > 0.70:
        return "high"
    if 0.45 <= score <= 0.55:
        return "low"
    return "medium"


async def _save_and_maybe_retrain(
    session,
    detection_id: str,
    features: list[float],
    label: int,
    label_source: str,
    text_excerpt: str | None,
) -> bool:
    """
    Persiste um TrainingExample e dispara retrain do RF se necessário.
    Retorna True se retrain foi disparado.
    """
    settings = get_settings()
    repo = TrainingRepository(session)

    await repo.create({
        "id": str(uuid.uuid4()),
        "detection_id": detection_id,
        "label": label,
        "label_source": label_source,
        "features_json": json.dumps(features),
        "text_excerpt": text_excerpt,
    })

    unused_count = await repo.count_unused()
    if unused_count >= settings.retrain_threshold:
        examples = await repo.get_all_for_training()
        await repo.mark_all_as_used()
        await learning_service.retrain_rf_with_examples(examples)
        return True
    return False


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------

@router.post("", response_model=DetectResponse, status_code=200)
async def detect(
    body: DetectRequest,
    session: DbSession,
    background_tasks: BackgroundTasks,
):
    """
    Analisa texto ou imagem e retorna veredicto de IA.

    **Fluxo para texto:**
    1. Heurísticas estatísticas
    2. Random Forest (se modelo disponível)
    3. Claude Haiku (se score na zona incerta 0.35–0.65 e ANTHROPIC_API_KEY configurada)

    **Fluxo para imagem:**
    Claude Sonnet Vision (sempre — heurísticas não se aplicam a imagens)

    O ID retornado pode ser enviado ao endpoint `/feedback` para
    corrigir o resultado e treinar o modelo.
    """
    settings = get_settings()
    detection_id = str(uuid.uuid4())
    start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Caminho: IMAGEM
    # -----------------------------------------------------------------------
    if body.image_base64:
        if not settings.anthropic_api_key:
            raise HTTPException(
                status_code=422,
                detail="Image analysis requires ANTHROPIC_API_KEY to be configured.",
            )

        try:
            result = await claude_service.analyze_image_with_claude(
                body.image_base64, body.image_media_type, settings.anthropic_api_key
            )
        except claude_service.ClaudeServiceError as exc:
            raise HTTPException(status_code=502, detail=f"Claude API error: {exc}")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return DetectResponse(
            id=detection_id,
            input_type="image",
            verdict=result["verdict"],
            ai_probability_score=result["probability_ai"],
            confidence_level=result["confidence"],
            claude_used=True,
            claude_explanation=result.get("explanation"),
            model_version="claude-vision-v1",
            processing_time_ms=elapsed_ms,
        )

    # -----------------------------------------------------------------------
    # Caminho: TEXTO
    # -----------------------------------------------------------------------
    text = body.text  # mypy: not None here (validated by schema)

    # Camada 1 + 2: heurísticas e RF
    cascade = detection_service.analyze_with_cascade(text)
    final_score = cascade["ai_probability_score"]
    features = cascade["features"]
    claude_used = False
    claude_explanation = None

    # Camada 3: Claude (zona incerta + API key disponível)
    if cascade["needs_claude"] and settings.anthropic_api_key:
        try:
            claude_result = await claude_service.analyze_text_with_claude(
                text, settings.anthropic_api_key
            )
            final_score = _blend_with_claude(final_score, claude_result["probability_ai"])
            claude_used = True
            claude_explanation = claude_result.get("explanation")
        except claude_service.ClaudeServiceError as exc:
            # Claude falhou → mantém resultado ML, não quebra o endpoint
            logger.warning("Claude fallback failed, using ML score: %s", exc)

    verdict = _score_to_verdict(final_score)
    confidence = _score_to_confidence(final_score)

    # Aprendizado contínuo: salva exemplo com label automático
    auto_label: int | None = 1 if verdict == "ai" else (0 if verdict == "human" else None)
    if auto_label is not None:
        text_excerpt = text[:300] if text else None
        background_tasks.add_task(
            _save_and_maybe_retrain,
            session,
            detection_id,
            features,
            auto_label,
            "auto",
            text_excerpt,
        )

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return DetectResponse(
        id=detection_id,
        input_type="text",
        verdict=verdict,
        ai_probability_score=final_score,
        confidence_level=confidence,
        heuristic_score=cascade["heuristic_score"],
        ml_score=cascade["ml_score"],
        claude_used=claude_used,
        claude_explanation=claude_explanation,
        model_version=cascade["model_version"],
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

@router.post("/feedback", response_model=FeedbackResponse, status_code=200)
async def feedback(
    body: FeedbackRequest,
    session: DbSession,
    background_tasks: BackgroundTasks,
):
    """
    Envia o rótulo correto para um resultado de detecção.

    O modelo aprende imediatamente via SGDClassifier.partial_fit().
    Quando RETRAIN_THRESHOLD exemplos confirmados se acumulam, o
    Random Forest é retreinado em background.
    """
    settings = get_settings()
    repo = TrainingRepository(session)

    # Busca o exemplo existente pelo detection_id
    existing = await repo.get_by_detection_id(body.detection_id)

    label_int = 1 if body.correct_label == "ai" else 0

    if existing is None:
        # Detecção não foi salva (ex: verdict era "uncertain") — cria agora sem features
        # Neste caso não temos features para partial_fit, apenas registramos o feedback
        await repo.create({
            "id": str(uuid.uuid4()),
            "detection_id": body.detection_id,
            "label": label_int,
            "label_source": "user_feedback",
            "features_json": "[]",  # vazio — não será usado em treino ML
            "text_excerpt": None,
        })
        return FeedbackResponse(
            detection_id=body.detection_id,
            label_accepted=body.correct_label,
            message="Feedback recorded. No feature vector available for this detection.",
            retrain_triggered=False,
        )

    # Atualiza o label do exemplo existente
    existing.label = label_int
    existing.label_source = "user_feedback"
    await session.commit()

    # SGD partial_fit imediato (se tiver features)
    features_raw = json.loads(existing.features_json) if existing.features_json else []
    if len(features_raw) == 5:
        background_tasks.add_task(
            learning_service.partial_fit_example, features_raw, label_int
        )

    # Verifica se precisa disparar retrain do RF
    unused_count = await repo.count_unused()
    retrain_triggered = False
    if unused_count >= settings.retrain_threshold:
        examples = await repo.get_all_for_training()
        await repo.mark_all_as_used()
        background_tasks.add_task(learning_service.retrain_rf_with_examples, examples)
        retrain_triggered = True

    return FeedbackResponse(
        detection_id=body.detection_id,
        label_accepted=body.correct_label,
        message=(
            "Feedback accepted. Model is learning from this correction."
            + (" Random Forest retrain triggered." if retrain_triggered else "")
        ),
        retrain_triggered=retrain_triggered,
    )
