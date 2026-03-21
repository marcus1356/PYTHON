"""
Testes unitários dos schemas Pydantic.

Ensino:
- Schemas são a primeira linha de defesa contra dados inválidos.
- Testar schemas separado da API é mais rápido e preciso.
- Nenhum banco de dados ou HTTP necessário.
"""

import pytest
from pydantic import ValidationError

from app.schemas.submission import SubmissionCreate, SubmissionUpdate


# ── SubmissionCreate ────────────────────────────────────────────────────────


def test_schema_create_valid():
    obj = SubmissionCreate(text_content="a" * 50)
    assert obj.text_content == "a" * 50
    assert obj.title is None
    assert obj.source is None


def test_schema_create_text_too_short():
    with pytest.raises(ValidationError) as exc_info:
        SubmissionCreate(text_content="curto")
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("text_content",) for e in errors)


def test_schema_create_text_too_long():
    with pytest.raises(ValidationError):
        SubmissionCreate(text_content="x" * 50_001)


def test_schema_create_text_exact_min():
    obj = SubmissionCreate(text_content="a" * 50)
    assert len(obj.text_content) == 50


def test_schema_create_text_exact_max():
    obj = SubmissionCreate(text_content="a" * 50_000)
    assert len(obj.text_content) == 50_000


def test_schema_create_title_max_length():
    with pytest.raises(ValidationError):
        SubmissionCreate(text_content="a" * 50, title="x" * 256)


def test_schema_create_source_max_length():
    with pytest.raises(ValidationError):
        SubmissionCreate(text_content="a" * 50, source="x" * 101)


def test_schema_create_title_within_limit():
    obj = SubmissionCreate(text_content="a" * 50, title="x" * 255)
    assert len(obj.title) == 255


def test_schema_create_source_within_limit():
    obj = SubmissionCreate(text_content="a" * 50, source="x" * 100)
    assert len(obj.source) == 100


# ── SubmissionUpdate ────────────────────────────────────────────────────────


def test_schema_update_title_only():
    obj = SubmissionUpdate(title="New Title")
    assert obj.title == "New Title"
    assert obj.source is None


def test_schema_update_source_only():
    obj = SubmissionUpdate(source="blog-post")
    assert obj.source == "blog-post"
    assert obj.title is None


def test_schema_update_both_fields():
    obj = SubmissionUpdate(title="T", source="S")
    assert obj.title == "T"
    assert obj.source == "S"


def test_schema_update_no_fields_raises():
    with pytest.raises(ValidationError) as exc_info:
        SubmissionUpdate()
    errors = exc_info.value.errors()
    # Deve ter um erro de validação informando que pelo menos um campo é necessário
    assert len(errors) >= 1


def test_schema_update_title_max_length():
    with pytest.raises(ValidationError):
        SubmissionUpdate(title="x" * 256)