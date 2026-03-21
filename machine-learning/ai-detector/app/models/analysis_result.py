import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    submission_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        unique=True,  # garante relação 1:1
        nullable=False,
    )

    # Resultado principal
    ai_probability_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_level: Mapped[str] = mapped_column(String(10), nullable=False)  # low|medium|high
    verdict: Mapped[str] = mapped_column(String(10), nullable=False)  # human|uncertain|ai

    # Heurísticas individuais armazenadas para auditabilidade e consulta
    avg_sentence_length: Mapped[float | None] = mapped_column(Float, nullable=True)
    vocabulary_richness: Mapped[float | None] = mapped_column(Float, nullable=True)
    burstiness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    punctuation_density: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_word_length: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Escape hatch: features adicionais em JSON sem precisar de migration
    features_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    submission: Mapped["Submission"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Submission",
        back_populates="analysis_result",
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisResult submission={self.submission_id!r} "
            f"score={self.ai_probability_score:.2f} verdict={self.verdict!r}>"
        )
