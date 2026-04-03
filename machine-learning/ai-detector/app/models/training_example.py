"""
Modelo de banco de dados para o buffer de aprendizado contínuo.

Cada DetectRequest que recebe um label (automático ou pelo usuário via /feedback)
gera um TrainingExample. Esses registros alimentam o partial_fit (SGDClassifier)
e o retrain periódico do Random Forest.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainingExample(Base):
    __tablename__ = "training_examples"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Link opcional para o registro de detecção que originou este exemplo
    detection_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # 0 = humano, 1 = IA
    label: Mapped[int] = mapped_column(Integer, nullable=False)

    # Como o label foi obtido: "auto" (do próprio modelo) | "user_feedback" | "dataset"
    label_source: Mapped[str] = mapped_column(String(20), nullable=False, default="auto")

    # Vetor de features serializado como JSON  ex: "[18.5, 0.42, 0.31, 0.05, 5.2]"
    features_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Trecho do texto para auditoria (primeiros 300 chars)
    text_excerpt: Mapped[str | None] = mapped_column(String(300), nullable=True)

    # Controle de uso em treinamento
    used_in_training: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index("ix_training_label", "label"),
        Index("ix_training_used", "used_in_training"),
        Index("ix_training_created_at", "created_at"),
        Index("ix_training_label_source", "label_source"),
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingExample id={self.id!r} label={self.label} "
            f"source={self.label_source!r} used={self.used_in_training}>"
        )
