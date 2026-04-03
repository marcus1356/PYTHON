"""
Wrapper para a API da Anthropic (Claude).

Responsabilidades:
  - Texto incerto: chama claude-haiku (rápido, barato) para desempate.
  - Imagem: chama claude-sonnet (Vision) para análise completa.

Princípio de design:
  - Retorna sempre um dict padronizado; o endpoint nunca vê objetos Anthropic.
  - Erros de API são capturados aqui e relançados como ClaudeServiceError
    para o endpoint decidir o que fazer (fallback ou 502).
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


class ClaudeServiceError(Exception):
    """Erro ao chamar a API da Anthropic."""


def _parse_json_response(raw: str) -> dict:
    """
    Extrai JSON da resposta do Claude, mesmo que venha com texto ao redor.
    Claude às vezes adiciona explicação antes/depois do bloco JSON.
    """
    # Tenta direto
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Tenta extrair bloco ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Tenta qualquer {…} no texto
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ClaudeServiceError(f"Could not parse JSON from Claude response: {raw[:200]}")


async def analyze_text_with_claude(text: str, api_key: str) -> dict:
    """
    Chama Claude Haiku para analisar se um texto foi escrito por IA.

    Usado quando o modelo ML retorna score na zona de incerteza (0.35–0.65).

    Retorna:
      {
        "verdict": "ai" | "human" | "uncertain",
        "confidence": "high" | "medium" | "low",
        "probability_ai": 0.0–1.0,
        "explanation": "..."
      }
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ClaudeServiceError("anthropic SDK not installed. Run: pip install anthropic") from exc

    client = AsyncAnthropic(api_key=api_key)

    # Trunca para não estourar o contexto nem gastar tokens desnecessários
    excerpt = text[:3000] if len(text) > 3000 else text

    prompt = (
        "You are an expert in detecting AI-generated text. "
        "Analyze the following text and determine if it was written by an AI or a human.\n\n"
        f"Text:\n{excerpt}\n\n"
        "Respond ONLY with a valid JSON object (no markdown, no extra text):\n"
        '{"verdict": "ai"|"human"|"uncertain", '
        '"confidence": "high"|"medium"|"low", '
        '"probability_ai": 0.0-1.0, '
        '"explanation": "one sentence reason"}'
    )

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        return _parse_json_response(raw)
    except ClaudeServiceError:
        raise
    except Exception as exc:
        raise ClaudeServiceError(f"Claude text API error: {exc}") from exc


async def analyze_image_with_claude(
    image_base64: str,
    media_type: str,
    api_key: str,
) -> dict:
    """
    Chama Claude Sonnet Vision para detectar se uma imagem é gerada por IA.

    Retorna o mesmo formato de analyze_text_with_claude.
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ClaudeServiceError("anthropic SDK not installed. Run: pip install anthropic") from exc

    client = AsyncAnthropic(api_key=api_key)

    prompt = (
        "Analyze this image carefully. Determine whether it appears to be AI-generated "
        "(e.g., from Midjourney, DALL-E, Stable Diffusion) or created/photographed by a human.\n\n"
        "Look for: unnatural textures, impossible geometry, garbled text, "
        "dreamlike or overly perfect lighting, uncanny faces.\n\n"
        "Respond ONLY with a valid JSON object (no markdown, no extra text):\n"
        '{"verdict": "ai"|"human"|"uncertain", '
        '"confidence": "high"|"medium"|"low", '
        '"probability_ai": 0.0-1.0, '
        '"explanation": "one sentence reason"}'
    )

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        raw = response.content[0].text
        return _parse_json_response(raw)
    except ClaudeServiceError:
        raise
    except Exception as exc:
        raise ClaudeServiceError(f"Claude Vision API error: {exc}") from exc
