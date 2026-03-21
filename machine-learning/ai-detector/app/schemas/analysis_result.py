from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class FeaturesDetail(BaseModel):
    avg_sentence_length: float | None
    vocabulary_richness: float | None
    burstiness_score: float | None
    punctuation_density: float | None
    avg_word_length: float | None


class AnalysisResultResponse(BaseModel):
    submission_id: str
    ai_probability_score: float = Field(ge=0.0, le=1.0)
    confidence_level: str
    verdict: str
    features: FeaturesDetail
    model_version: str
    processing_time_ms: int | None
    analyzed_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def build_features_from_orm(cls, data: Any) -> Any:
        """
        Quando validado a partir de um ORM object (from_attributes),
        constrói o campo `features` composto a partir das colunas individuais.
        """
        if isinstance(data, dict):
            return data
        # ORM object: mapeia colunas individuais para o campo aninhado `features`
        return {
            "submission_id": data.submission_id,
            "ai_probability_score": data.ai_probability_score,
            "confidence_level": data.confidence_level,
            "verdict": data.verdict,
            "features": {
                "avg_sentence_length": data.avg_sentence_length,
                "vocabulary_richness": data.vocabulary_richness,
                "burstiness_score": data.burstiness_score,
                "punctuation_density": data.punctuation_density,
                "avg_word_length": data.avg_word_length,
            },
            "model_version": data.model_version,
            "processing_time_ms": data.processing_time_ms,
            "analyzed_at": data.analyzed_at,
        }

    @classmethod
    def from_orm_model(cls, obj: object) -> "AnalysisResultResponse":
        return cls.model_validate(obj)
