"""
Testes unitários da camada de serviço (SubmissionService).

Ensino:
- Testa a lógica de negócio diretamente, sem HTTP.
- Mais rápido que testes de integração e mais preciso para isolar bugs.
- Usa o db_session do conftest (banco em memória, limpo a cada teste).
"""

import pytest

from app.core.exceptions import (
    AnalysisResultNotFoundError,
    SubmissionAlreadyAnalyzedError,
    SubmissionNotFoundError,
)
from app.schemas.submission import SubmissionCreate, SubmissionUpdate
from app.services.submission_service import SubmissionService

pytestmark = pytest.mark.asyncio

LONG_TEXT = (
    "Artificial intelligence has profoundly transformed modern industries and research fields. "
    "Machine learning models now process vast amounts of data with remarkable accuracy. "
    "Natural language processing enables computers to understand human communication effectively. "
    "These advancements create both tremendous opportunities and significant challenges for society."
)


# ── create ─────────────────────────────────────────────────────────────────


async def test_service_create_calculates_word_count(db_session):
    service = SubmissionService(db_session)
    data = SubmissionCreate(text_content=LONG_TEXT, title="Test", source="pytest")
    result = await service.create(data)

    expected_words = len(LONG_TEXT.split())
    assert result.word_count == expected_words


async def test_service_create_calculates_char_count(db_session):
    service = SubmissionService(db_session)
    data = SubmissionCreate(text_content=LONG_TEXT)
    result = await service.create(data)

    assert result.char_count == len(LONG_TEXT)


async def test_service_create_status_is_pending(db_session):
    service = SubmissionService(db_session)
    data = SubmissionCreate(text_content=LONG_TEXT)
    result = await service.create(data)

    assert result.status == "pending"


async def test_service_create_optional_fields_none(db_session):
    service = SubmissionService(db_session)
    data = SubmissionCreate(text_content=LONG_TEXT)
    result = await service.create(data)

    assert result.title is None
    assert result.source is None


# ── get_detail ──────────────────────────────────────────────────────────────


async def test_service_get_detail_returns_text_content(db_session):
    service = SubmissionService(db_session)
    created = await service.create(SubmissionCreate(text_content=LONG_TEXT))

    detail = await service.get_detail(created.id)
    assert detail.text_content == LONG_TEXT


async def test_service_get_detail_not_found_raises(db_session):
    service = SubmissionService(db_session)
    with pytest.raises(SubmissionNotFoundError):
        await service.get_detail("id-que-nao-existe")


# ── list_paginated ──────────────────────────────────────────────────────────


async def test_service_list_empty(db_session):
    service = SubmissionService(db_session)
    result = await service.list_paginated(page=1, page_size=10, status=None, source=None)

    assert result.total == 0
    assert result.items == []


async def test_service_list_returns_created(db_session):
    service = SubmissionService(db_session)
    await service.create(SubmissionCreate(text_content=LONG_TEXT, source="blog"))
    result = await service.list_paginated(page=1, page_size=10, status=None, source=None)

    assert result.total == 1


async def test_service_list_filter_by_status(db_session):
    service = SubmissionService(db_session)
    await service.create(SubmissionCreate(text_content=LONG_TEXT))

    pending = await service.list_paginated(page=1, page_size=10, status="pending", source=None)
    completed = await service.list_paginated(page=1, page_size=10, status="completed", source=None)

    assert pending.total == 1
    assert completed.total == 0


async def test_service_list_filter_by_source(db_session):
    service = SubmissionService(db_session)
    await service.create(SubmissionCreate(text_content=LONG_TEXT, source="blog"))
    await service.create(SubmissionCreate(text_content=LONG_TEXT, source="essay"))

    result = await service.list_paginated(page=1, page_size=10, status=None, source="blog")
    assert result.total == 1
    assert result.items[0].source == "blog"


async def test_service_list_pagination(db_session):
    service = SubmissionService(db_session)
    for _ in range(5):
        await service.create(SubmissionCreate(text_content=LONG_TEXT))

    page1 = await service.list_paginated(page=1, page_size=3, status=None, source=None)
    page2 = await service.list_paginated(page=2, page_size=3, status=None, source=None)

    assert len(page1.items) == 3
    assert len(page2.items) == 2
    assert page1.total == 5
    assert page1.pages == 2


# ── update ──────────────────────────────────────────────────────────────────


async def test_service_update_title(db_session):
    service = SubmissionService(db_session)
    created = await service.create(SubmissionCreate(text_content=LONG_TEXT, title="Old"))

    updated = await service.update(created.id, SubmissionUpdate(title="New Title"))
    assert updated.title == "New Title"


async def test_service_update_source(db_session):
    service = SubmissionService(db_session)
    created = await service.create(SubmissionCreate(text_content=LONG_TEXT))

    updated = await service.update(created.id, SubmissionUpdate(source="new-source"))
    assert updated.source == "new-source"


async def test_service_update_not_found_raises(db_session):
    service = SubmissionService(db_session)
    with pytest.raises(SubmissionNotFoundError):
        await service.update("id-invalido", SubmissionUpdate(title="X"))


# ── delete ──────────────────────────────────────────────────────────────────


async def test_service_delete_removes_submission(db_session):
    service = SubmissionService(db_session)
    created = await service.create(SubmissionCreate(text_content=LONG_TEXT))

    await service.delete(created.id)

    with pytest.raises(SubmissionNotFoundError):
        await service.get_detail(created.id)


async def test_service_delete_not_found_raises(db_session):
    service = SubmissionService(db_session)
    with pytest.raises(SubmissionNotFoundError):
        await service.delete("id-invalido")


# ── analyze ─────────────────────────────────────────────────────────────────


async def test_service_analyze_sets_status_completed(db_session):
    service = SubmissionService(db_session)
    created = await service.create(SubmissionCreate(text_content=LONG_TEXT))

    await service.analyze(created.id)
    detail = await service.get_detail(created.id)
    assert detail.status == "completed"


async def test_service_analyze_creates_result(db_session):
    service = SubmissionService(db_session)
    created = await service.create(SubmissionCreate(text_content=LONG_TEXT))

    await service.analyze(created.id)
    detail = await service.get_detail(created.id)

    assert detail.analysis_result is not None
    assert 0.0 <= detail.analysis_result.ai_probability_score <= 1.0


async def test_service_analyze_already_analyzed_raises(db_session):
    service = SubmissionService(db_session)
    created = await service.create(SubmissionCreate(text_content=LONG_TEXT))
    await service.analyze(created.id)

    with pytest.raises(SubmissionAlreadyAnalyzedError):
        await service.analyze(created.id)


async def test_service_analyze_force_reanalyze(db_session):
    service = SubmissionService(db_session)
    created = await service.create(SubmissionCreate(text_content=LONG_TEXT))
    await service.analyze(created.id)

    # Não deve lançar exceção
    await service.analyze(created.id, force=True)
    detail = await service.get_detail(created.id)
    assert detail.status == "completed"


async def test_service_analyze_not_found_raises(db_session):
    service = SubmissionService(db_session)
    with pytest.raises(SubmissionNotFoundError):
        await service.analyze("id-invalido")