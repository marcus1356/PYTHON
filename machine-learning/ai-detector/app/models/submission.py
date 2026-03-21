import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Máquina de estados: pending → analyzing → completed | failed
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")

    submitted_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relacionamento 1:1 com AnalysisResult
    # lazy="selectin" é obrigatório em async — evita lazy load síncrono
    analysis_result: Mapped["AnalysisResult"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "AnalysisResult",
        back_populates="submission",
        uselist=False,
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_submissions_status", "status"),
        Index("ix_submissions_created_at", "created_at"),
        Index("ix_submissions_source", "source"),
    )

    def __repr__(self) -> str:
        return f"<Submission id={self.id!r} status={self.status!r}>"
