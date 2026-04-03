"""
Schemas Pydantic para o endpoint unificado /detect e /feedback.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    """
    Entrada do endpoint POST /detect.
    Aceita texto ou imagem (base64) — exatamente um dos dois é obrigatório.
    """

    text: str | None = Field(
        default=None,
        min_length=50,
        max_length=50_000,
        description="Texto a ser analisado (50–50 000 caracteres)",
    )
    image_base64: str | None = Field(
        default=None,
        description="Imagem codificada em Base64 (JPEG, PNG, GIF, WEBP)",
    )
    image_media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = Field(
        default="image/jpeg",
        description="MIME type da imagem",
    )
    source: str | None = Field(
        default=None,
        max_length=100,
        description="Origem do conteúdo (ex: 'student-essay', 'social-media')",
    )

    @model_validator(mode="after")
    def one_of_text_or_image(self) -> "DetectRequest":
        if self.text is None and self.image_base64 is None:
            raise ValueError("Provide either 'text' or 'image_base64'.")
        if self.text is not None and self.image_base64 is not None:
            raise ValueError("Provide only one of 'text' or 'image_base64', not both.")
        return self


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class DetectResponse(BaseModel):
    """Resultado completo de uma análise de detecção."""

    id: str = Field(description="UUID único desta detecção (use em /feedback)")
    input_type: Literal["text", "image"]

    # Resultado final
    verdict: Literal["ai", "human", "uncertain"]
    ai_probability_score: float = Field(ge=0.0, le=1.0)
    confidence_level: Literal["high", "medium", "low"]

    # Camadas usadas
    heuristic_score: float | None = None
    ml_score: float | None = None
    claude_used: bool = False
    claude_explanation: str | None = None

    # Meta
    model_version: str
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """Correção enviada pelo usuário para um resultado de detecção."""

    detection_id: str = Field(description="ID retornado pelo /detect")
    correct_label: Literal["ai", "human"] = Field(
        description="Rótulo correto: 'ai' ou 'human'"
    )


class FeedbackResponse(BaseModel):
    detection_id: str
    label_accepted: Literal["ai", "human"]
    message: str
    retrain_triggered: bool = False
