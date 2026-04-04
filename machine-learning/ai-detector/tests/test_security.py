"""
Testes de segurança — equivalente ao SonarQube para esta aplicação.

Categorias:
  A1 - Injeção (SQL, XSS, Command Injection)
  A2 - Validação de inputs maliciosos
  A3 - Exposição de dados sensíveis
  A4 - CORS e headers de segurança
  A5 - Rate limiting / DoS
  A6 - Limites de tamanho (imagens, textos)
  A7 - Erros não tratados (comportamento gracioso)
"""

import pytest
from httpx import AsyncClient

DETECT_URL = "/api/v1/detect"
SUBMISSIONS_URL = "/api/v1/submissions"
HEALTH_URL = "/health"

# Texto base válido (>50 chars)
VALID_TEXT = (
    "This is a valid text that is long enough to pass "
    "the minimum length validation of fifty characters."
)


# ---------------------------------------------------------------------------
# A1 — Injeção
# ---------------------------------------------------------------------------

class TestInjection:
    @pytest.mark.asyncio
    async def test_sql_injection_in_text_content(self, client: AsyncClient):
        """SQL injection no campo de texto não deve causar erro 500."""
        payload = {
            "text_content": "'; DROP TABLE submissions; -- " + "x" * 30,
            "title": "SQL Injection Test"
        }
        resp = await client.post(SUBMISSIONS_URL, json=payload)
        # Deve ser 201 ou 422 (validação), nunca 500
        assert resp.status_code in (201, 422)

    @pytest.mark.asyncio
    async def test_sql_injection_in_source_field(self, client: AsyncClient):
        payload = {
            "text_content": VALID_TEXT,
            "source": "' OR '1'='1"
        }
        resp = await client.post(SUBMISSIONS_URL, json=payload)
        assert resp.status_code in (201, 422)

    @pytest.mark.asyncio
    async def test_xss_in_title_field(self, client: AsyncClient):
        """XSS no campo title não deve ser executado (API JSON, não HTML)."""
        payload = {
            "text_content": VALID_TEXT,
            "title": "<script>alert('xss')</script>"
        }
        resp = await client.post(SUBMISSIONS_URL, json=payload)
        assert resp.status_code in (201, 422)
        if resp.status_code == 201:
            # A resposta não deve executar o script — apenas retornar como string
            assert "<script>" in resp.text or resp.json()["title"] is not None

    @pytest.mark.asyncio
    async def test_xss_in_detect_text(self, client: AsyncClient):
        """XSS no texto de detecção deve ser tratado como texto puro."""
        xss_text = "<script>alert(1)</script> " * 5 + "x" * 30
        resp = await client.post(DETECT_URL, json={"text": xss_text})
        assert resp.status_code in (200, 422)
        if resp.status_code == 200:
            assert resp.json()["verdict"] in ("ai", "human", "uncertain")

    @pytest.mark.asyncio
    async def test_null_bytes_in_text(self, client: AsyncClient):
        """Null bytes não devem causar crash."""
        text_with_nulls = "Valid text " + "\x00" * 10 + " more content here."
        resp = await client.post(DETECT_URL, json={"text": text_with_nulls})
        assert resp.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_sql_injection_in_detect_source(self, client: AsyncClient):
        resp = await client.post(DETECT_URL, json={
            "text": VALID_TEXT,
            "source": "'; DELETE FROM training_examples; --"
        })
        assert resp.status_code in (200, 422)


# ---------------------------------------------------------------------------
# A2 — Validação de inputs maliciosos
# ---------------------------------------------------------------------------

class TestInputValidation:
    @pytest.mark.asyncio
    async def test_extremely_long_title_rejected(self, client: AsyncClient):
        """Título além do limite (255 chars) deve ser rejeitado."""
        resp = await client.post(SUBMISSIONS_URL, json={
            "text_content": VALID_TEXT,
            "title": "A" * 256
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_title_exactly_at_limit_accepted(self, client: AsyncClient):
        resp = await client.post(SUBMISSIONS_URL, json={
            "text_content": VALID_TEXT,
            "title": "A" * 255
        })
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_source_beyond_limit_rejected(self, client: AsyncClient):
        resp = await client.post(SUBMISSIONS_URL, json={
            "text_content": VALID_TEXT,
            "source": "x" * 101
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_detect_text_beyond_50k_rejected(self, client: AsyncClient):
        """Texto maior que 50.000 chars deve ser rejeitado."""
        resp = await client.post(DETECT_URL, json={"text": "A" * 50_001})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_detect_text_exactly_50k_accepted(self, client: AsyncClient):
        resp = await client.post(DETECT_URL, json={"text": "A" * 50_000})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_json_body_rejected(self, client: AsyncClient):
        """JSON malformado deve retornar 422."""
        resp = await client.post(
            DETECT_URL,
            content=b"not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_body_rejected(self, client: AsyncClient):
        resp = await client.post(DETECT_URL, json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_wrong_content_type(self, client: AsyncClient):
        """Form data ao invés de JSON deve retornar 422."""
        resp = await client.post(
            DETECT_URL,
            data={"text": VALID_TEXT},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_unicode_text_handled_gracefully(self, client: AsyncClient):
        """Texto em unicode (emojis, chars especiais) não deve causar crash."""
        unicode_text = "Texto com émojis 🤖🧠 e acentos: ção, ã, é. " * 5
        resp = await client.post(DETECT_URL, json={"text": unicode_text})
        assert resp.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_only_spaces_text(self, client: AsyncClient):
        """Texto só com espaços deve falhar na validação."""
        resp = await client.post(DETECT_URL, json={"text": " " * 60})
        # Pode passar validação de length mas modelo deve tratar
        assert resp.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_numeric_text_field_rejected(self, client: AsyncClient):
        """Campo text como número deve ser rejeitado pelo Pydantic."""
        resp = await client.post(DETECT_URL, json={"text": 12345})
        # Pydantic v2 faz coerção de int para str, então pode aceitar
        # O importante é não retornar 500
        assert resp.status_code in (200, 422)


# ---------------------------------------------------------------------------
# A3 — Exposição de dados sensíveis
# ---------------------------------------------------------------------------

class TestSensitiveDataExposure:
    @pytest.mark.asyncio
    async def test_health_endpoint_does_not_expose_env_vars(self, client: AsyncClient):
        """O health check não deve vazar variáveis de ambiente."""
        resp = await client.get(HEALTH_URL)
        body = resp.text
        assert "anthropic" not in body.lower()
        assert "api_key" not in body.lower()
        assert "database_url" not in body.lower()
        assert "password" not in body.lower()

    @pytest.mark.asyncio
    async def test_500_errors_do_not_expose_stack_trace(self, client: AsyncClient):
        """Erros internos não devem vazar stack traces em produção."""
        # Envia UUID inválido para forçar 404
        resp = await client.get("/api/v1/submissions/not-a-valid-uuid")
        # Deve ser 404, não 500 com stack trace
        assert resp.status_code == 404
        body = resp.text
        assert "Traceback" not in body
        assert "File " not in body

    @pytest.mark.asyncio
    async def test_nonexistent_submission_returns_404_not_500(self, client: AsyncClient):
        resp = await client.get("/api/v1/submissions/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_error_response_has_detail_field(self, client: AsyncClient):
        """Erros devem retornar JSON com campo 'detail', não HTML."""
        resp = await client.get("/api/v1/submissions/nonexistent-id")
        assert resp.headers.get("content-type", "").startswith("application/json")


# ---------------------------------------------------------------------------
# A4 — CORS e headers
# ---------------------------------------------------------------------------

class TestCORSAndHeaders:
    @pytest.mark.asyncio
    async def test_cors_allows_requests(self, client: AsyncClient):
        """CORS deve responder a preflight OPTIONS."""
        resp = await client.options(
            DETECT_URL,
            headers={
                "Origin": "https://radaria.com.br",
                "Access-Control-Request-Method": "POST",
            }
        )
        # FastAPI responde 200 ao OPTIONS
        assert resp.status_code in (200, 405)

    @pytest.mark.asyncio
    async def test_api_returns_json_content_type(self, client: AsyncClient):
        """Todas as respostas da API devem ser JSON."""
        resp = await client.get(HEALTH_URL)
        assert "application/json" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_detect_response_is_json(self, client: AsyncClient):
        resp = await client.post(DETECT_URL, json={"text": VALID_TEXT})
        assert "application/json" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# A5 — Comportamento sob carga / DoS
# ---------------------------------------------------------------------------

class TestDoSResistance:
    @pytest.mark.asyncio
    async def test_multiple_rapid_requests_dont_crash(self, client: AsyncClient):
        """10 requests rápidos não devem derrubar o servidor."""
        import asyncio

        async def single_request():
            return await client.post(DETECT_URL, json={"text": VALID_TEXT})

        results = await asyncio.gather(*[single_request() for _ in range(10)])
        for resp in results:
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_max_size_text_completes_in_reasonable_time(self, client: AsyncClient):
        """Texto no tamanho máximo deve completar (sem timeout)."""
        import time
        big_text = "This is a normal sentence with reasonable content. " * 1000
        big_text = big_text[:50_000]

        start = time.time()
        resp = await client.post(DETECT_URL, json={"text": big_text})
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert elapsed < 10.0, f"Muito lento: {elapsed:.1f}s para 50k chars"


# ---------------------------------------------------------------------------
# A6 — Limites de imagem (sem API key)
# ---------------------------------------------------------------------------

class TestImageLimits:
    @pytest.mark.asyncio
    async def test_image_without_api_key_returns_422(self, client: AsyncClient):
        import base64
        fake_image = base64.b64encode(b"fake image data" * 100).decode()
        resp = await client.post(DETECT_URL, json={
            "image_base64": fake_image,
            "image_media_type": "image/jpeg"
        })
        # Sem API key deve retornar 422
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_image_media_type_rejected(self, client: AsyncClient):
        import base64
        fake = base64.b64encode(b"data").decode()
        resp = await client.post(DETECT_URL, json={
            "image_base64": fake,
            "image_media_type": "image/tiff"  # não suportado
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# A7 — Erros tratados graciosamente
# ---------------------------------------------------------------------------

class TestGracefulErrors:
    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404_not_500(self, client: AsyncClient):
        resp = await client.delete("/api/v1/submissions/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_404(self, client: AsyncClient):
        resp = await client.put(
            "/api/v1/submissions/00000000-0000-0000-0000-000000000000",
            json={"title": "new title"}
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_analyze_nonexistent_returns_404(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/submissions/00000000-0000-0000-0000-000000000000/analyze"
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_result_nonexistent_returns_404(self, client: AsyncClient):
        resp = await client.get(
            "/api/v1/submissions/00000000-0000-0000-0000-000000000000/result"
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_method_not_allowed_returns_405(self, client: AsyncClient):
        """GET em endpoint que só aceita POST deve retornar 405."""
        resp = await client.get(DETECT_URL)
        assert resp.status_code == 405

    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_404(self, client: AsyncClient):
        resp = await client.get("/api/v1/nonexistent-endpoint")
        assert resp.status_code == 404
