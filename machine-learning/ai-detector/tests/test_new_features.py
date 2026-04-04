"""
Testes das 7 novas features da Fase 3 do detection_service.

Cada feature tem comportamento esperado documentado com base em
pesquisa linguística (Goh & Barabasi 2008, Lei de Zipf, etc.).
"""

import pytest

from app.services.detection_service import (
    FEATURE_NAMES,
    N_FEATURES_V3,
    analyze_with_cascade,
    compute_avg_word_length,
    compute_bigram_repetition_score,
    compute_first_person_ratio,
    compute_hapax_legomena_ratio,
    compute_hedge_word_ratio,
    compute_lexical_diversity_entropy,
    compute_question_density,
    compute_transition_word_density,
    extract_features,
)

# ---------------------------------------------------------------------------
# Textos de referência
# ---------------------------------------------------------------------------

AI_TEXT = (
    "Artificial intelligence has fundamentally transformed the landscape of modern technology. "
    "Furthermore, the systematic application of machine learning algorithms enables unprecedented "
    "analytical capabilities. Moreover, neural network architectures facilitate sophisticated "
    "pattern recognition. It is important to note that these advancements have significant "
    "implications for various industries. Consequently, organizations must adapt accordingly. "
    "Additionally, investment in AI research yields substantial returns for stakeholders."
)

HUMAN_TEXT = (
    "I was really struggling with this problem all week. Maybe I'm overthinking it? "
    "My colleague suggested a different approach, and honestly, I think she might be right. "
    "I tried it yesterday and it sort of worked? Not perfectly, but better. "
    "Perhaps if I tweak a few things it'll click. I'm not sure yet, but I feel "
    "like we're getting closer. Does anyone else deal with this kind of thing?"
)

REPETITIVE_TEXT = (
    "the cat sat on the mat the cat sat on the mat the cat sat on the mat "
    "the cat sat on the mat the cat sat on the mat the cat sat on the mat "
    "the cat sat on the mat the cat sat on the mat the cat sat on the mat "
    "the cat sat on the mat the cat sat on the mat the cat sat on the mat "
    "big brown fox jumped over the lazy dog and the cat sat on the mat again."
)


# ---------------------------------------------------------------------------
# Feature 5: transition_word_density
# ---------------------------------------------------------------------------

class TestTransitionWordDensity:
    def test_ai_text_has_high_density(self):
        score = compute_transition_word_density(AI_TEXT)
        assert score > 0.02, f"Esperado > 0.02, obtido {score}"

    def test_human_text_has_low_density(self):
        score = compute_transition_word_density(HUMAN_TEXT)
        assert score < 0.02, f"Esperado < 0.02, obtido {score}"

    def test_returns_float(self):
        assert isinstance(compute_transition_word_density(AI_TEXT), float)

    def test_empty_text_returns_zero(self):
        assert compute_transition_word_density("") == 0.0

    def test_score_between_0_and_1(self):
        score = compute_transition_word_density(AI_TEXT)
        assert 0.0 <= score <= 1.0

    def test_furthermore_detected(self):
        text = "Furthermore, " + "a" * 40 + " " + "b" * 10
        score = compute_transition_word_density(text)
        assert score > 0.0

    def test_moreover_detected(self):
        text = "Moreover, the results are significant. " * 5
        score = compute_transition_word_density(text)
        assert score > 0.0


# ---------------------------------------------------------------------------
# Feature 6: first_person_ratio
# ---------------------------------------------------------------------------

class TestFirstPersonRatio:
    def test_human_text_has_high_ratio(self):
        score = compute_first_person_ratio(HUMAN_TEXT)
        assert score > 0.01, f"Esperado > 0.01, obtido {score}"

    def test_ai_text_has_low_ratio(self):
        score = compute_first_person_ratio(AI_TEXT)
        assert score < 0.01, f"Esperado < 0.01, obtido {score}"

    def test_text_with_many_i_pronouns(self):
        text = "I think I should go. I believe I can. I feel I will succeed. " * 3
        score = compute_first_person_ratio(text)
        assert score > 0.05

    def test_empty_text_returns_zero(self):
        assert compute_first_person_ratio("") == 0.0

    def test_score_between_0_and_1(self):
        assert 0.0 <= compute_first_person_ratio(HUMAN_TEXT) <= 1.0

    def test_we_pronoun_counted(self):
        text = "We decided to proceed. We thought it was best. We agreed." * 3
        score = compute_first_person_ratio(text)
        assert score > 0.05


# ---------------------------------------------------------------------------
# Feature 7: hedge_word_ratio
# ---------------------------------------------------------------------------

class TestHedgeWordRatio:
    def test_human_text_has_hedging(self):
        score = compute_hedge_word_ratio(HUMAN_TEXT)
        assert score > 0.005, f"Esperado > 0.005, obtido {score}"

    def test_ai_text_has_less_hedging(self):
        score = compute_hedge_word_ratio(AI_TEXT)
        assert score < 0.01, f"Esperado < 0.01, obtido {score}"

    def test_maybe_detected(self):
        text = "Maybe it works. Perhaps it doesn't. " * 5
        score = compute_hedge_word_ratio(text)
        assert score > 0.0

    def test_empty_text_returns_zero(self):
        assert compute_hedge_word_ratio("") == 0.0

    def test_score_non_negative(self):
        assert compute_hedge_word_ratio(AI_TEXT) >= 0.0


# ---------------------------------------------------------------------------
# Feature 8: question_density
# ---------------------------------------------------------------------------

class TestQuestionDensity:
    def test_human_text_has_questions(self):
        score = compute_question_density(HUMAN_TEXT)
        assert score > 0.0, "Texto humano deveria ter perguntas"

    def test_ai_text_has_no_questions(self):
        score = compute_question_density(AI_TEXT)
        assert score == 0.0, f"Texto de IA não deveria ter perguntas, obtido {score}"

    def test_multiple_questions(self):
        text = "What is this? How does it work? Why is it so? When did it start?"
        score = compute_question_density(text)
        assert score > 0.5

    def test_empty_text_returns_zero(self):
        assert compute_question_density("") == 0.0

    def test_no_questions_returns_zero(self):
        text = "This is a statement. Another statement here. And one more."
        assert compute_question_density(text) == 0.0


# ---------------------------------------------------------------------------
# Feature 9: bigram_repetition_score
# ---------------------------------------------------------------------------

class TestBigramRepetitionScore:
    def test_repetitive_text_has_high_score(self):
        score = compute_bigram_repetition_score(REPETITIVE_TEXT)
        assert score > 0.3, f"Esperado > 0.3, obtido {score}"

    def test_diverse_text_has_lower_score(self):
        score = compute_bigram_repetition_score(HUMAN_TEXT)
        assert score < 0.5

    def test_very_short_text_returns_zero(self):
        assert compute_bigram_repetition_score("hi") == 0.0

    def test_score_between_0_and_1(self):
        score = compute_bigram_repetition_score(AI_TEXT)
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        assert isinstance(compute_bigram_repetition_score(AI_TEXT), float)


# ---------------------------------------------------------------------------
# Feature 10: lexical_diversity_entropy
# ---------------------------------------------------------------------------

class TestLexicalDiversityEntropy:
    def test_diverse_text_has_high_entropy(self):
        score = compute_lexical_diversity_entropy(HUMAN_TEXT)
        assert score > 0.7, f"Esperado > 0.7, obtido {score}"

    def test_repetitive_text_has_lower_entropy(self):
        score_repetitive = compute_lexical_diversity_entropy(REPETITIVE_TEXT)
        score_diverse = compute_lexical_diversity_entropy(HUMAN_TEXT)
        assert score_repetitive < score_diverse

    def test_score_between_0_and_1(self):
        score = compute_lexical_diversity_entropy(AI_TEXT)
        assert 0.0 <= score <= 1.0

    def test_empty_text_returns_zero(self):
        assert compute_lexical_diversity_entropy("") == 0.0

    def test_single_word_returns_zero(self):
        assert compute_lexical_diversity_entropy("hello") == 0.0

    def test_returns_float(self):
        assert isinstance(compute_lexical_diversity_entropy(AI_TEXT), float)


# ---------------------------------------------------------------------------
# Feature 11: hapax_legomena_ratio
# ---------------------------------------------------------------------------

class TestHapaxLegomenaRatio:
    def test_diverse_text_has_high_hapax(self):
        score = compute_hapax_legomena_ratio(HUMAN_TEXT)
        assert score > 0.4, f"Esperado > 0.4, obtido {score}"

    def test_repetitive_text_has_lower_hapax_than_diverse(self):
        """Texto repetitivo deve ter MENOS hapax que texto diverso."""
        score_repetitive = compute_hapax_legomena_ratio(REPETITIVE_TEXT)
        score_diverse = compute_hapax_legomena_ratio(HUMAN_TEXT)
        assert score_repetitive < score_diverse, (
            f"Repetitivo ({score_repetitive:.2f}) deveria ter menos hapax "
            f"que diverso ({score_diverse:.2f})"
        )

    def test_score_between_0_and_1(self):
        score = compute_hapax_legomena_ratio(AI_TEXT)
        assert 0.0 <= score <= 1.0

    def test_empty_text_returns_zero(self):
        assert compute_hapax_legomena_ratio("") == 0.0

    def test_all_unique_words_returns_one(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        score = compute_hapax_legomena_ratio(text)
        assert score == 1.0


# ---------------------------------------------------------------------------
# extract_features — vetor completo
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_12_features(self):
        features = extract_features(AI_TEXT)
        assert len(features) == N_FEATURES_V3 == 12

    def test_returns_5_features_when_requested(self):
        features = extract_features(AI_TEXT, n_features=5)
        assert len(features) == 5

    def test_all_values_are_floats(self):
        features = extract_features(AI_TEXT)
        for i, f in enumerate(features):
            assert isinstance(f, float), f"Feature {i} não é float: {type(f)}"

    def test_all_values_non_negative(self):
        features = extract_features(AI_TEXT)
        for i, f in enumerate(features):
            assert f >= 0.0, f"Feature {i} é negativa: {f}"

    def test_feature_names_count_matches(self):
        assert len(FEATURE_NAMES) == N_FEATURES_V3

    def test_empty_text_does_not_crash(self):
        features = extract_features("")
        assert len(features) == N_FEATURES_V3

    def test_short_text_does_not_crash(self):
        features = extract_features("Hi there, how are you doing today?")
        assert len(features) == N_FEATURES_V3

    def test_unicode_text_does_not_crash(self):
        text = "Isso é um texto em português com acentuação. " * 5
        features = extract_features(text)
        assert len(features) == N_FEATURES_V3


# ---------------------------------------------------------------------------
# analyze_with_cascade — integração completa
# ---------------------------------------------------------------------------

class TestAnalyzeWithCascade:
    def test_returns_required_keys(self):
        result = analyze_with_cascade(AI_TEXT)
        required = [
            "ai_probability_score", "confidence_level", "verdict",
            "needs_claude", "heuristic_score", "ml_score",
            "features", "named_features", "model_version", "processing_time_ms"
        ]
        for key in required:
            assert key in result, f"Chave ausente: {key}"

    def test_score_between_0_and_1(self):
        result = analyze_with_cascade(AI_TEXT)
        assert 0.0 <= result["ai_probability_score"] <= 1.0

    def test_heuristic_score_between_0_and_1(self):
        result = analyze_with_cascade(AI_TEXT)
        assert 0.0 <= result["heuristic_score"] <= 1.0

    def test_features_list_has_12_elements(self):
        result = analyze_with_cascade(AI_TEXT)
        assert len(result["features"]) == 12

    def test_named_features_has_12_keys(self):
        result = analyze_with_cascade(AI_TEXT)
        assert len(result["named_features"]) == 12

    def test_named_features_keys_match_feature_names(self):
        result = analyze_with_cascade(AI_TEXT)
        for name in FEATURE_NAMES:
            assert name in result["named_features"], f"Feature ausente: {name}"

    def test_needs_claude_is_bool(self):
        result = analyze_with_cascade(AI_TEXT)
        assert isinstance(result["needs_claude"], bool)

    def test_ai_text_tends_toward_ai_verdict(self):
        result = analyze_with_cascade(AI_TEXT)
        # Com RF treinado, texto de IA típico deve ter score alto
        assert result["ai_probability_score"] > 0.4

    def test_human_text_tends_toward_human_verdict(self):
        result = analyze_with_cascade(HUMAN_TEXT)
        assert result["ai_probability_score"] < 0.7

    def test_processing_time_positive(self):
        result = analyze_with_cascade(AI_TEXT)
        assert result["processing_time_ms"] >= 0

    def test_model_version_is_string(self):
        result = analyze_with_cascade(AI_TEXT)
        assert isinstance(result["model_version"], str)
        assert len(result["model_version"]) > 0
