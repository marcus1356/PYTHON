import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_create_submission(client: AsyncClient, submission_payload: dict):
    response = await client.post("/api/v1/submissions", json=submission_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "pending"
    assert data["word_count"] > 0
    assert "id" in data
    assert "text_content" not in data  # texto não aparece no resumo


async def test_create_submission_text_too_short(client: AsyncClient):
    response = await client.post(
        "/api/v1/submissions",
        json={"text_content": "curto demais"},
    )
    assert response.status_code == 422


async def test_list_submissions_empty(client: AsyncClient):
    response = await client.get("/api/v1/submissions")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []


async def test_list_submissions_with_data(client: AsyncClient, submission_payload: dict):
    await client.post("/api/v1/submissions", json=submission_payload)
    response = await client.get("/api/v1/submissions")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1


async def test_get_submission_detail(client: AsyncClient, submission_payload: dict):
    create_resp = await client.post("/api/v1/submissions", json=submission_payload)
    sub_id = create_resp.json()["id"]

    response = await client.get(f"/api/v1/submissions/{sub_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == sub_id
    assert "text_content" in data
    assert data["analysis_result"] is None


async def test_get_submission_not_found(client: AsyncClient):
    response = await client.get("/api/v1/submissions/nonexistent-id")
    assert response.status_code == 404


async def test_update_submission(client: AsyncClient, submission_payload: dict):
    create_resp = await client.post("/api/v1/submissions", json=submission_payload)
    sub_id = create_resp.json()["id"]

    response = await client.put(
        f"/api/v1/submissions/{sub_id}",
        json={"title": "Updated Title"},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Updated Title"


async def test_update_submission_no_fields(client: AsyncClient, submission_payload: dict):
    create_resp = await client.post("/api/v1/submissions", json=submission_payload)
    sub_id = create_resp.json()["id"]
    response = await client.put(f"/api/v1/submissions/{sub_id}", json={})
    assert response.status_code == 422


async def test_delete_submission(client: AsyncClient, submission_payload: dict):
    create_resp = await client.post("/api/v1/submissions", json=submission_payload)
    sub_id = create_resp.json()["id"]

    delete_resp = await client.delete(f"/api/v1/submissions/{sub_id}")
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/api/v1/submissions/{sub_id}")
    assert get_resp.status_code == 404


async def test_delete_nonexistent(client: AsyncClient):
    response = await client.delete("/api/v1/submissions/nonexistent-id")
    assert response.status_code == 404


async def test_filter_by_status(client: AsyncClient, submission_payload: dict):
    await client.post("/api/v1/submissions", json=submission_payload)
    response = await client.get("/api/v1/submissions?status=pending")
    assert response.status_code == 200
    assert response.json()["total"] == 1

    response = await client.get("/api/v1/submissions?status=completed")
    assert response.status_code == 200
    assert response.json()["total"] == 0
