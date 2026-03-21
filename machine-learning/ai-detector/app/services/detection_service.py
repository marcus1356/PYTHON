"""
Motor de detecção de IA — Fase 1: Heurísticas estatísticas.

Ensino:
- Cada heurística é uma função pura (sem efeitos colaterais, sem dependências externas).
- Funções puras são 100% testáveis sem banco de dados ou servidor rodando.
- Na Fase 2, este módulo será substituído por um modelo ML real
  sem alterar nenhum router ou service de negócio.
"""

import re
import statistics
import time

MODEL_VERSION = "heuristic-v1"


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text.lower())


def compute_avg_sentence_length(text: str) -> float:
    """
    Heurística 1: Comprimento médio das frases (em palavras).
    IA tende a produzir frases uniformes entre 18–26 palavras.
    Humanos variam muito mais.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    lengths = [len(_tokenize(s)) for s in sentences]
    return sum(lengths) / len(lengths)


def compute_vocabulary_richness(text: str) -> float:
    """
    Heurística 2: Type-Token Ratio (TTR) = palavras únicas / total de palavras.
    IA tende a TTR entre 0.35–0.55 (vocabulário moderado e consistente).
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_burstiness(text: str) -> float:
    """
    Heurística 3: Coeficiente de Variação (CV) dos comprimentos de frases.
    CV = desvio padrão / média.
    Humanos variam muito (CV alto). IA é estranhamente uniforme (CV baixo).
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return 1.0  # texto curto demais para calcular — assume humano
    lengths = [len(_tokenize(s)) for s in sentences]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0
    std = statistics.stdev(lengths)
    return std / mean


def compute_punctuation_density(text: str) -> float:
    """
    Heurística 4: Proporção de caracteres de pontuação no texto.
    IA usa pontuação de forma moderada e previsível (0.03–0.07).
    """
    if not text:
        return 0.0
    punct_count = sum(1 for c in text if c in ".,;:!?\"'()-–—")
    return punct_count / len(text)


def compute_avg_word_length(text: str) -> float:
    """
    Heurística 5: Comprimento médio das palavras (em caracteres).
    IA tende a usar vocabulário levemente mais elaborado (avg > 5.0 chars).
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _compute_score(
    avg_sentence_len: float,
    vocabulary_richness: float,
    burstiness: float,
    punct_density: float,
    avg_word_len: float,
) -> float:
    """
    Combina as heurísticas em um score final entre 0.0 e 1.0.
    Cada heurística contribui com um peso máximo.
    """
    score = 0.0

    # Frases uniformemente longas → sinal de IA
    if 18.0 <= avg_sentence_len <= 26.0:
        score += 0.20

    # TTR moderado → sinal de IA
    if 0.35 <= vocabulary_richness <= 0.55:
        score += 0.20

    # Baixa variação no comprimento das frases → sinal de IA
    if burstiness < 0.40:
        score += 0.25

    # Densidade de pontuação previsível → sinal de IA
    if 0.03 <= punct_density <= 0.07:
        score += 0.15

    # Palavras levemente mais longas → sinal de IA
    if avg_word_len > 5.0:
        score += 0.20

    return min(score, 1.0)


def _compute_confidence(score: float) -> str:
    if score < 0.30 or score > 0.70:
        return "high"
    if 0.45 <= score <= 0.55:
        return "low"
    return "medium"


def _compute_verdict(score: float) -> str:
    if score >= 0.60:
        return "ai"
    if score <= 0.40:
        return "human"
    return "uncertain"


def analyze_text(text: str) -> dict:
    """
    Ponto de entrada principal do detector.
    Retorna um dicionário com todos os dados necessários para criar AnalysisResult.
    """
    start = time.perf_counter()

    avg_sentence_len = compute_avg_sentence_length(text)
    vocab_richness = compute_vocabulary_richness(text)
    burstiness = compute_burstiness(text)
    punct_density = compute_punctuation_density(text)
    avg_word_len = compute_avg_word_length(text)

    score = _compute_score(
        avg_sentence_len,
        vocab_richness,
        burstiness,
        punct_density,
        avg_word_len,
    )

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "ai_probability_score": round(score, 4),
        "confidence_level": _compute_confidence(score),
        "verdict": _compute_verdict(score),
        "avg_sentence_length": round(avg_sentence_len, 4),
        "vocabulary_richness": round(vocab_richness, 4),
        "burstiness_score": round(burstiness, 4),
        "punctuation_density": round(punct_density, 4),
        "avg_word_length": round(avg_word_len, 4),
        "model_version": MODEL_VERSION,
        "processing_time_ms": elapsed_ms,
    }
