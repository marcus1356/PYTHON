import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def _create_submission(client: AsyncClient, payload: dict) -> str:
    resp = await client.post("/api/v1/submissions", json=payload)
    assert resp.status_code == 201
    return resp.json()["id"]


async def test_analyze_submission(client: AsyncClient, submission_payload: dict):
    sub_id = await _create_submission(client, submission_payload)

    response = await client.post(f"/api/v1/submissions/{sub_id}/analyze")
    assert response.status_code == 202
    data = response.json()
    assert data["submission_id"] == sub_id
    assert data["status"] == "completed"


async def test_get_result_after_analysis(client: AsyncClient, submission_payload: dict):
    sub_id = await _create_submission(client, submission_payload)
    await client.post(f"/api/v1/submissions/{sub_id}/analyze")

    response = await client.get(f"/api/v1/submissions/{sub_id}/result")
    assert response.status_code == 200
    data = response.json()
    assert data["submission_id"] == sub_id
    assert 0.0 <= data["ai_probability_score"] <= 1.0
    assert data["verdict"] in ("human", "uncertain", "ai")
    assert data["confidence_level"] in ("low", "medium", "high")
    assert "features" in data
    assert data["model_version"] == "heuristic-v1"


async def test_get_result_without_analysis(client: AsyncClient, submission_payload: dict):
    sub_id = await _create_submission(client, submission_payload)
    response = await client.get(f"/api/v1/submissions/{sub_id}/result")
    assert response.status_code == 404


async def test_analyze_already_analyzed_conflict(client: AsyncClient, submission_payload: dict):
    sub_id = await _create_submission(client, submission_payload)
    await client.post(f"/api/v1/submissions/{sub_id}/analyze")

    response = await client.post(f"/api/v1/submissions/{sub_id}/analyze")
    assert response.status_code == 409


async def test_force_reanalyze(client: AsyncClient, submission_payload: dict):
    sub_id = await _create_submission(client, submission_payload)
    await client.post(f"/api/v1/submissions/{sub_id}/analyze")

    response = await client.post(
        f"/api/v1/submissions/{sub_id}/analyze?force_reanalyze=true"
    )
    assert response.status_code == 202


async def test_analyze_nonexistent_submission(client: AsyncClient):
    response = await client.post("/api/v1/submissions/nonexistent-id/analyze")
    assert response.status_code == 404


async def test_submission_detail_includes_result(client: AsyncClient, submission_payload: dict):
    sub_id = await _create_submission(client, submission_payload)
    await client.post(f"/api/v1/submissions/{sub_id}/analyze")

    response = await client.get(f"/api/v1/submissions/{sub_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["analysis_result"] is not None
    assert data["analysis_result"]["verdict"] in ("human", "uncertain", "ai")
