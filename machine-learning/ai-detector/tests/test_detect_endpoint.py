"""
Testes do endpoint POST /api/v1/detect e POST /api/v1/detect/feedback.

Cobertura:
- Inputs válidos (texto)
- Validação de schema (texto curto, ambos campos, nenhum campo)
- Campos obrigatórios na resposta
- Ranges de valores (score 0-1, campos Literal)
- Feedback com detection_id existente e inexistente
- Imagem sem API key (deve retornar 422)
- Mock do Claude para testar o fallback
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient

# Texto curto demais (< 50 chars)
SHORT_TEXT = "Too short."

# Texto longo o suficiente (> 50 chars, claramente humano)
HUMAN_TEXT = (
    "I was walking home last Tuesday when it started raining. "
    "Grabbed my jacket, but too late — already soaked. "
    "I laughed at myself. What was I thinking? "
    "My friend texted asking how my day went. Wet. Very wet."
)

# Texto longo com padrões de IA
AI_TEXT = (
    "Artificial intelligence has fundamentally transformed the landscape of modern technology. "
    "Furthermore, the systematic application of machine learning algorithms enables unprecedented "
    "analytical capabilities across diverse domains. Moreover, the integration of neural network "
    "architectures facilitates sophisticated pattern recognition. It is important to note that "
    "these advancements have significant implications for various industries and stakeholders."
)

DETECT_URL = "/api/v1/detect"
FEEDBACK_URL = "/api/v1/detect/feedback"


# ---------------------------------------------------------------------------
# Testes de input válido — texto
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_text_returns_200(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_detect_text_response_has_required_fields(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    data = resp.json()
    required = ["id", "input_type", "verdict", "ai_probability_score",
                "confidence_level", "model_version", "processing_time_ms"]
    for field in required:
        assert field in data, f"Campo ausente: {field}"


@pytest.mark.asyncio
async def test_detect_text_input_type_is_text(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    assert resp.json()["input_type"] == "text"


@pytest.mark.asyncio
async def test_detect_text_score_between_0_and_1(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    score = resp.json()["ai_probability_score"]
    assert 0.0 <= score <= 1.0, f"Score fora do range: {score}"


@pytest.mark.asyncio
async def test_detect_text_verdict_valid_values(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    assert resp.json()["verdict"] in ("ai", "human", "uncertain")


@pytest.mark.asyncio
async def test_detect_text_confidence_valid_values(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    assert resp.json()["confidence_level"] in ("high", "medium", "low")


@pytest.mark.asyncio
async def test_detect_text_processing_time_positive(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    assert resp.json()["processing_time_ms"] >= 0


@pytest.mark.asyncio
async def test_detect_text_returns_uuid_id(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    detection_id = resp.json()["id"]
    # Valida formato UUID (8-4-4-4-12)
    import re
    assert re.match(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        detection_id
    ), f"ID não é UUID: {detection_id}"


@pytest.mark.asyncio
async def test_detect_text_claude_used_false_without_key(client: AsyncClient):
    """Sem ANTHROPIC_API_KEY, Claude nunca deve ser chamado."""
    resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    assert resp.json()["claude_used"] is False


@pytest.mark.asyncio
async def test_detect_text_with_source_metadata(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={
        "text": AI_TEXT,
        "source": "pytest-test"
    })
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Testes de validação de schema
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_text_too_short_returns_422(client: AsyncClient):
    resp = await client.post(DETECT_URL, json={"text": SHORT_TEXT})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_detect_no_input_returns_422(client: AsyncClient):
    """Nem texto nem imagem — deve falhar."""
    resp = await client.post(DETECT_URL, json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_detect_both_text_and_image_returns_422(client: AsyncClient):
    """Texto e imagem juntos — deve falhar."""
    resp = await client.post(DETECT_URL, json={
        "text": AI_TEXT,
        "image_base64": "abc123"
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_detect_text_exactly_50_chars(client: AsyncClient):
    """Exatamente no limite mínimo deve passar."""
    text_50 = "a" * 50
    resp = await client.post(DETECT_URL, json={"text": text_50})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_detect_text_49_chars_fails(client: AsyncClient):
    """49 chars deve falhar."""
    resp = await client.post(DETECT_URL, json={"text": "a" * 49})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_detect_text_too_long_returns_422(client: AsyncClient):
    """Texto maior que 50.000 chars deve falhar."""
    resp = await client.post(DETECT_URL, json={"text": "a" * 50_001})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_detect_invalid_image_media_type_returns_422(client: AsyncClient):
    """Media type inválido deve falhar."""
    resp = await client.post(DETECT_URL, json={
        "image_base64": "abc",
        "image_media_type": "image/bmp"  # não suportado
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Testes de imagem
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_image_without_api_key_returns_422(client: AsyncClient):
    """Imagem sem ANTHROPIC_API_KEY deve retornar 422."""
    import base64
    # Imagem PNG mínima (1x1 pixel)
    png_1x1 = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
        b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
        b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode()

    resp = await client.post(DETECT_URL, json={
        "image_base64": png_1x1,
        "image_media_type": "image/png"
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Testes com mock do Claude
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_text_calls_claude_when_uncertain(client: AsyncClient):
    """
    Quando o modelo retorna score na zona incerta e há API key,
    Claude deve ser chamado.
    """
    uncertain_result = {
        "ai_probability_score": 0.50,
        "confidence_level": "low",
        "verdict": "uncertain",
        "needs_claude": True,
        "heuristic_score": 0.50,
        "ml_score": None,
        "features": [15.0, 0.45, 0.35, 0.05, 5.0, 0.01, 0.01, 0.01, 0.01, 0.10, 0.80, 0.50],
        "named_features": {},
        "model_version": "cascade-v1",
        "processing_time_ms": 5,
    }
    claude_result = {
        "verdict": "ai",
        "confidence": "high",
        "probability_ai": 0.85,
        "explanation": "Uses formal transition words and lacks hedging."
    }

    with patch("app.api.v1.endpoints.detect.detection_service.analyze_with_cascade",
               return_value=uncertain_result), \
         patch("app.api.v1.endpoints.detect.get_settings") as mock_settings, \
         patch("app.api.v1.endpoints.detect.claude_service.analyze_text_with_claude",
               new_callable=AsyncMock, return_value=claude_result) as mock_claude:

        mock_settings.return_value.anthropic_api_key = "sk-test-fake-key"
        mock_settings.return_value.retrain_threshold = 50

        resp = await client.post(DETECT_URL, json={"text": AI_TEXT})

    assert resp.status_code == 200
    data = resp.json()
    assert data["claude_used"] is True
    assert data["claude_explanation"] is not None
    mock_claude.assert_called_once()


@pytest.mark.asyncio
async def test_detect_text_claude_failure_falls_back_to_ml(client: AsyncClient):
    """Se Claude falhar, o endpoint deve retornar o score ML sem quebrar."""
    from app.services.claude_service import ClaudeServiceError

    uncertain_result = {
        "ai_probability_score": 0.50,
        "confidence_level": "low",
        "verdict": "uncertain",
        "needs_claude": True,
        "heuristic_score": 0.50,
        "ml_score": None,
        "features": [15.0, 0.45, 0.35, 0.05, 5.0, 0.01, 0.01, 0.01, 0.01, 0.10, 0.80, 0.50],
        "named_features": {},
        "model_version": "cascade-v1",
        "processing_time_ms": 5,
    }

    with patch("app.api.v1.endpoints.detect.detection_service.analyze_with_cascade",
               return_value=uncertain_result), \
         patch("app.api.v1.endpoints.detect.get_settings") as mock_settings, \
         patch("app.api.v1.endpoints.detect.claude_service.analyze_text_with_claude",
               new_callable=AsyncMock,
               side_effect=ClaudeServiceError("API timeout")):

        mock_settings.return_value.anthropic_api_key = "sk-test-fake-key"
        mock_settings.return_value.retrain_threshold = 50

        resp = await client.post(DETECT_URL, json={"text": AI_TEXT})

    # Não deve quebrar — deve retornar resultado ML
    assert resp.status_code == 200
    data = resp.json()
    assert data["claude_used"] is False


# ---------------------------------------------------------------------------
# Testes do endpoint /feedback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feedback_with_existing_detection(client: AsyncClient):
    """Feedback para detection_id existente deve retornar 200."""
    # Primeiro cria uma detecção
    detect_resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    detection_id = detect_resp.json()["id"]

    resp = await client.post(FEEDBACK_URL, json={
        "detection_id": detection_id,
        "correct_label": "ai"
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_feedback_with_nonexistent_detection(client: AsyncClient):
    """Feedback para detection_id inexistente ainda deve retornar 200."""
    resp = await client.post(FEEDBACK_URL, json={
        "detection_id": "00000000-0000-0000-0000-000000000000",
        "correct_label": "human"
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_feedback_response_has_required_fields(client: AsyncClient):
    detect_resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    detection_id = detect_resp.json()["id"]

    resp = await client.post(FEEDBACK_URL, json={
        "detection_id": detection_id,
        "correct_label": "human"
    })
    data = resp.json()
    for field in ["detection_id", "label_accepted", "message", "retrain_triggered"]:
        assert field in data, f"Campo ausente: {field}"


@pytest.mark.asyncio
async def test_feedback_label_accepted_matches_input(client: AsyncClient):
    detect_resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    detection_id = detect_resp.json()["id"]

    for label in ("ai", "human"):
        resp = await client.post(FEEDBACK_URL, json={
            "detection_id": detection_id,
            "correct_label": label
        })
        assert resp.json()["label_accepted"] == label


@pytest.mark.asyncio
async def test_feedback_invalid_label_returns_422(client: AsyncClient):
    resp = await client.post(FEEDBACK_URL, json={
        "detection_id": "any-id",
        "correct_label": "maybe"  # inválido
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feedback_missing_detection_id_returns_422(client: AsyncClient):
    resp = await client.post(FEEDBACK_URL, json={"correct_label": "ai"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feedback_missing_label_returns_422(client: AsyncClient):
    resp = await client.post(FEEDBACK_URL, json={"detection_id": "some-id"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feedback_retrain_triggered_false_below_threshold(client: AsyncClient):
    """Com menos de 50 exemplos, retrain_triggered deve ser False."""
    detect_resp = await client.post(DETECT_URL, json={"text": AI_TEXT})
    detection_id = detect_resp.json()["id"]

    resp = await client.post(FEEDBACK_URL, json={
        "detection_id": detection_id,
        "correct_label": "ai"
    })
    # Abaixo do threshold de 50 exemplos
    assert resp.json()["retrain_triggered"] is False
