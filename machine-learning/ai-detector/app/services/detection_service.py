"""
Motor de detecção de IA — Cascade: Heurísticas → RF → Claude (opcional).

Arquitetura em camadas:
  Camada 1 — Heurísticas estatísticas (sem dependências externas, sem I/O)
  Camada 2 — Random Forest treinado no dataset HC3 (joblib, in-process)
  Camada 3 — Claude API (assíncrono, chamado pelo endpoint quando necessário)

Separação de responsabilidades:
  - Este módulo: heurísticas + ML síncrono.
  - claude_service.py: chamadas assíncronas à API da Anthropic.
  - O endpoint /detect orquestra os três níveis.
"""

import logging
import re
import statistics
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_VERSION = "heuristic-v1"
CASCADE_VERSION = "cascade-v1"

_ML_DIR = Path(__file__).resolve().parent.parent / "models" / "ml"
_RF_MODEL_PATH = _ML_DIR / "detector_hc3_rf.joblib"

# Cache em memória para não recarregar o modelo a cada requisição
_rf_model = None


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text.lower())


# ---------------------------------------------------------------------------
# Heurísticas (Camada 1) — funções puras, testáveis sem infra
# ---------------------------------------------------------------------------

def compute_avg_sentence_length(text: str) -> float:
    """
    Heurística 1: Comprimento médio das frases (em palavras).
    IA tende a produzir frases uniformes entre 18–26 palavras.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    lengths = [len(_tokenize(s)) for s in sentences]
    return sum(lengths) / len(lengths)


def compute_vocabulary_richness(text: str) -> float:
    """
    Heurística 2: Type-Token Ratio (TTR) = palavras únicas / total.
    IA tende a TTR entre 0.35–0.55.
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_burstiness(text: str) -> float:
    """
    Heurística 3: CV dos comprimentos de frases.
    Humanos variam muito (CV alto). IA é estranhamente uniforme (CV baixo).
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return 1.0
    lengths = [len(_tokenize(s)) for s in sentences]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0
    std = statistics.stdev(lengths)
    return std / mean


def compute_punctuation_density(text: str) -> float:
    """Heurística 4: Proporção de caracteres de pontuação."""
    if not text:
        return 0.0
    punct_count = sum(1 for c in text if c in ".,;:!?\"'()-–—")
    return punct_count / len(text)


def compute_avg_word_length(text: str) -> float:
    """Heurística 5: Comprimento médio das palavras (em chars)."""
    words = _tokenize(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def extract_features(text: str) -> list[float]:
    """
    Extrai o vetor de features na mesma ordem usada para treinar o RF.
    Retorna: [avg_sentence_len, vocab_richness, burstiness, punct_density, avg_word_len]
    """
    return [
        compute_avg_sentence_length(text),
        compute_vocabulary_richness(text),
        compute_burstiness(text),
        compute_punctuation_density(text),
        compute_avg_word_length(text),
    ]


# ---------------------------------------------------------------------------
# Pontuação e veredito
# ---------------------------------------------------------------------------

def _heuristic_score(features: list[float]) -> float:
    avg_sl, vocab, burst, punct, avg_wl = features
    score = 0.0
    if 18.0 <= avg_sl <= 26.0:
        score += 0.20
    if 0.35 <= vocab <= 0.55:
        score += 0.20
    if burst < 0.40:
        score += 0.25
    if 0.03 <= punct <= 0.07:
        score += 0.15
    if avg_wl > 5.0:
        score += 0.20
    return min(score, 1.0)


def _verdict(score: float) -> str:
    if score >= 0.60:
        return "ai"
    if score <= 0.40:
        return "human"
    return "uncertain"


def _confidence(score: float) -> str:
    if score < 0.30 or score > 0.70:
        return "high"
    if 0.45 <= score <= 0.55:
        return "low"
    return "medium"


# ---------------------------------------------------------------------------
# Camada 2 — Random Forest
# ---------------------------------------------------------------------------

def _load_rf():
    """Carrega o modelo RF com cache em memória (singleton)."""
    global _rf_model
    if _rf_model is None and _RF_MODEL_PATH.exists():
        try:
            import joblib
            _rf_model = joblib.load(_RF_MODEL_PATH)
            logger.info("RF model loaded: %s", _RF_MODEL_PATH.name)
        except Exception as exc:
            logger.warning("Failed to load RF model: %s", exc)
    return _rf_model


def _rf_probability(features: list[float]) -> float | None:
    """Retorna probabilidade de IA do RF, ou None se modelo não disponível."""
    model = _load_rf()
    if model is None:
        return None
    try:
        import numpy as np
        prob = model.predict_proba(np.array([features]))[0][1]
        return float(prob)
    except Exception as exc:
        logger.warning("RF inference failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# API pública — Camada 1 (legada, mantida para SubmissionService)
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> dict:
    """
    Análise heurística pura. Mantida por compatibilidade com SubmissionService.
    Para novos usos, prefira analyze_with_cascade().
    """
    start = time.perf_counter()
    features = extract_features(text)
    avg_sl, vocab, burst, punct, avg_wl = features
    score = _heuristic_score(features)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "ai_probability_score": round(score, 4),
        "confidence_level": _confidence(score),
        "verdict": _verdict(score),
        "avg_sentence_length": round(avg_sl, 4),
        "vocabulary_richness": round(vocab, 4),
        "burstiness_score": round(burst, 4),
        "punctuation_density": round(punct, 4),
        "avg_word_length": round(avg_wl, 4),
        "model_version": MODEL_VERSION,
        "processing_time_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# API pública — Cascade (Camada 1 + 2, sem Claude)
# ---------------------------------------------------------------------------

def analyze_with_cascade(text: str) -> dict:
    """
    Analisa texto combinando heurísticas (Camada 1) e RF (Camada 2).

    Retorna dict com:
      - ai_probability_score: score final (0–1)
      - confidence_level: "high" | "medium" | "low"
      - verdict: "ai" | "human" | "uncertain"
      - needs_claude: bool — True quando score na zona incerta
      - heuristic_score: score bruto das heurísticas
      - ml_score: probabilidade do RF (None se modelo indisponível)
      - features: lista de 5 floats para aprendizado contínuo
      - model_version: identificador da versão
      - processing_time_ms: latência em ms
    """
    start = time.perf_counter()

    features = extract_features(text)
    h_score = _heuristic_score(features)
    rf_prob = _rf_probability(features)

    # Blend: se RF disponível, dá peso 70% RF + 30% heurística
    if rf_prob is not None:
        final_score = round(0.30 * h_score + 0.70 * rf_prob, 4)
        used_version = "cascade-rf-v1"
    else:
        final_score = round(h_score, 4)
        used_version = CASCADE_VERSION

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Zona incerta: |score - 0.5| < 0.15  →  [0.35, 0.65]
    needs_claude = abs(final_score - 0.5) < 0.15

    return {
        "ai_probability_score": final_score,
        "confidence_level": _confidence(final_score),
        "verdict": _verdict(final_score),
        "needs_claude": needs_claude,
        "heuristic_score": round(h_score, 4),
        "ml_score": round(rf_prob, 4) if rf_prob is not None else None,
        "features": [round(f, 4) for f in features],
        "model_version": used_version,
        "processing_time_ms": elapsed_ms,
    }
