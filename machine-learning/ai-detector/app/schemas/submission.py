from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class SubmissionCreate(BaseModel):
    text_content: str = Field(
        min_length=50,
        max_length=50_000,
        description="Texto a ser analisado (50–50.000 caracteres)",
    )
    title: str | None = Field(default=None, max_length=255)
    source: str | None = Field(
        default=None,
        max_length=100,
        description="Origem do texto (ex: 'student-essay', 'blog-post')",
    )


class SubmissionUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    source: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def at_least_one_field(self) -> "SubmissionUpdate":
        if self.title is None and self.source is None:
            raise ValueError("Provide at least one field to update (title or source).")
        return self


class SubmissionSummary(BaseModel):
    id: str
    title: str | None
    source: str | None
    word_count: int
    char_count: int
    status: str
    submitted_by: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SubmissionDetail(SubmissionSummary):
    text_content: str
    analysis_result: "AnalysisResultResponse | None" = None  # noqa: F821

    model_config = {"from_attributes": True}


# Import circular resolvido com update_forward_refs
from app.schemas.analysis_result import AnalysisResultResponse  # noqa: E402

SubmissionDetail.model_rebuild()
