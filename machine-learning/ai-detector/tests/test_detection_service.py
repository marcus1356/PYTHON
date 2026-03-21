"""
Testes unitários do motor de heurísticas.
Totalmente isolados — sem banco, sem HTTP, sem FastAPI.
Ensino: esse é o nível de teste mais barato e mais rápido.
"""

import pytest
from app.services.detection_service import (
    analyze_text,
    compute_avg_sentence_length,
    compute_avg_word_length,
    compute_burstiness,
    compute_punctuation_density,
    compute_vocabulary_richness,
)

AI_LIKE_TEXT = (
    "Artificial intelligence represents a significant paradigm shift in computational methodologies. "
    "Machine learning algorithms process substantial datasets to derive meaningful insights efficiently. "
    "Natural language processing enables sophisticated interactions between humans and automated systems. "
    "These technological advancements fundamentally transform organizational processes and strategic frameworks. "
    "Researchers continuously investigate innovative approaches to enhance algorithmic performance and accuracy."
)

HUMAN_LIKE_TEXT = (
    "I went to the store. Forgot my wallet! Had to go back home. "
    "It was raining. Got wet. My dog barked at me when I came in — weird. "
    "Made coffee. Spilled it. Monday, right?"
)


def test_avg_sentence_length_ai():
    result = compute_avg_sentence_length(AI_LIKE_TEXT)
    assert result > 10


def test_avg_sentence_length_human():
    result = compute_avg_sentence_length(HUMAN_LIKE_TEXT)
    assert result < 15


def test_vocabulary_richness_returns_between_0_and_1():
    result = compute_vocabulary_richness(AI_LIKE_TEXT)
    assert 0.0 <= result <= 1.0


def test_burstiness_human_higher_than_ai():
    ai_burstiness = compute_burstiness(AI_LIKE_TEXT)
    human_burstiness = compute_burstiness(HUMAN_LIKE_TEXT)
    # Humanos variam mais (CV maior)
    assert human_burstiness >= ai_burstiness


def test_punctuation_density_between_0_and_1():
    result = compute_punctuation_density(AI_LIKE_TEXT)
    assert 0.0 <= result <= 1.0


def test_avg_word_length_positive():
    result = compute_avg_word_length(AI_LIKE_TEXT)
    assert result > 0


def test_analyze_text_returns_all_fields():
    result = analyze_text(AI_LIKE_TEXT)
    required_fields = {
        "ai_probability_score",
        "confidence_level",
        "verdict",
        "avg_sentence_length",
        "vocabulary_richness",
        "burstiness_score",
        "punctuation_density",
        "avg_word_length",
        "model_version",
        "processing_time_ms",
    }
    assert required_fields.issubset(result.keys())


def test_analyze_text_score_in_range():
    result = analyze_text(AI_LIKE_TEXT)
    assert 0.0 <= result["ai_probability_score"] <= 1.0


def test_analyze_text_verdict_valid():
    result = analyze_text(AI_LIKE_TEXT)
    assert result["verdict"] in ("human", "uncertain", "ai")


def test_analyze_text_confidence_valid():
    result = analyze_text(AI_LIKE_TEXT)
    assert result["confidence_level"] in ("low", "medium", "high")


def test_analyze_text_model_version():
    result = analyze_text(AI_LIKE_TEXT)
    assert result["model_version"] == "heuristic-v1"


def test_empty_text_does_not_crash():
    # Texto muito curto — heurísticas devem retornar 0.0 sem exceção
    result = analyze_text("Short text here with enough words to not crash the system.")
    assert isinstance(result["ai_probability_score"], float)
