"""
Motor de detecção de IA — Cascade: Heurísticas → RF → Claude (opcional).

Arquitetura em camadas:
  Camada 1 — 12 features estatísticas (sem dependências externas, sem I/O)
  Camada 2 — Random Forest treinado no dataset HC3 (joblib, in-process)
  Camada 3 — Claude API (assíncrono, chamado pelo endpoint quando necessário)

Por que este modelo supera detectores baseados em LLM:
  LLMs alucinam — podem errar com alta confiança.
  Nosso modelo é determinístico, auditável, explicável e não tem custo por chamada.
  Claude é usado APENAS como desempate na zona de incerteza, não como núcleo.

Referências técnicas:
  - Mitchell et al. (2023) "DetectGPT" — perturbation-based detection
  - Goh & Barabasi (2008) — burstiness em linguagem humana
  - Zipf's Law + entropia de Shannon aplicada a n-gramas
  - Solaiman et al. (2019) — log-perplexity para detecção zero-shot
"""

import logging
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_VERSION = "heuristic-v1"
CASCADE_VERSION = "cascade-v1"

_ML_DIR = Path(__file__).resolve().parent.parent / "models" / "ml"
_RF_V3_PATH = _ML_DIR / "detector_v3_rf.joblib"      # modelo com 12 features
_RF_HC3_PATH = _ML_DIR / "detector_hc3_rf.joblib"    # fallback com 5 features

# Cache em memória
_rf_model = None
_rf_n_features: int = 0  # quantas features o modelo carregado espera

# ---------------------------------------------------------------------------
# Palavras características por categoria
# ---------------------------------------------------------------------------

# Palavras de transição que LLMs usam em excesso
_TRANSITION_WORDS = {
    "furthermore", "moreover", "additionally", "consequently", "therefore",
    "nevertheless", "nonetheless", "subsequently", "accordingly", "thus",
    "hence", "likewise", "similarly", "conversely", "alternatively",
    "specifically", "notably", "importantly", "significantly", "essentially",
    "ultimately", "comprehensively", "effectively", "efficiently",
    "in conclusion", "in summary", "to summarize", "in addition",
    "as a result", "it is worth noting", "it is important to note",
    "it should be noted", "it is essential", "it is crucial",
    "a wide range", "a variety of", "in terms of", "with regard to",
    "in order to", "due to the fact", "for the purpose of",
}

# Pronomes de primeira pessoa (humanos usam mais)
_FIRST_PERSON = {"i", "my", "me", "myself", "we", "our", "us", "ourselves", "mine", "ours"}

# Palavras de hedging (humanos hesitam mais)
_HEDGE_WORDS = {
    "maybe", "perhaps", "probably", "possibly", "might", "could", "seems",
    "appears", "roughly", "approximately", "somewhat", "rather", "fairly",
    "quite", "sort of", "kind of", "i think", "i believe", "i feel",
    "i suppose", "i guess", "i imagine", "not sure", "uncertain",
    "arguably", "presumably", "ostensibly",
}

# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text.lower())


def _bigrams(tokens: list[str]) -> list[tuple]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


# ---------------------------------------------------------------------------
# FEATURE GROUP 1: Features originais (Fase 1)
# ---------------------------------------------------------------------------

def compute_avg_sentence_length(text: str) -> float:
    """
    Comprimento médio das frases em palavras.
    IA tende a frases uniformes entre 18–26 palavras.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    lengths = [len(_tokenize(s)) for s in sentences]
    return sum(lengths) / len(lengths)


def compute_vocabulary_richness(text: str) -> float:
    """
    Type-Token Ratio: palavras únicas / total.
    IA tem TTR moderado e consistente (0.35–0.55).
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_burstiness(text: str) -> float:
    """
    Coeficiente de Variação (CV) dos comprimentos de frases.
    Linguagem humana é "bursty" (CV alto). IA é uniforme (CV baixo).
    Ref: Goh & Barabasi (2008).
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return 1.0
    lengths = [len(_tokenize(s)) for s in sentences]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0
    return statistics.stdev(lengths) / mean


def compute_punctuation_density(text: str) -> float:
    """Proporção de caracteres de pontuação."""
    if not text:
        return 0.0
    punct_count = sum(1 for c in text if c in ".,;:!?\"'()-–—")
    return punct_count / len(text)


def compute_avg_word_length(text: str) -> float:
    """Comprimento médio das palavras em caracteres. IA usa vocabulário levemente mais elaborado."""
    words = _tokenize(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


# ---------------------------------------------------------------------------
# FEATURE GROUP 2: Features avançadas (Fase 3)
# ---------------------------------------------------------------------------

def compute_transition_word_density(text: str) -> float:
    """
    Proporção de palavras/expressões de transição no texto.

    LLMs abusam de conectivos formais ('Furthermore', 'Moreover', 'Additionally').
    Humanos variam mais e usam contrações informais ('also', 'but', 'so').

    Sinal forte de IA: valor alto (> 0.04 em proporção de tokens).
    """
    text_lower = text.lower()
    words = _tokenize(text_lower)
    if not words:
        return 0.0

    count = 0
    # Expressões multipalavra (verificar no texto direto)
    for phrase in _TRANSITION_WORDS:
        if " " in phrase:
            count += text_lower.count(phrase)
        else:
            count += words.count(phrase)

    return count / len(words)


def compute_first_person_ratio(text: str) -> float:
    """
    Proporção de pronomes de primeira pessoa.

    Humanos fazem afirmações pessoais, expressam opiniões com 'I think', 'I feel'.
    LLMs evitam primeira pessoa para parecer objetivos e imparciais.

    Sinal de humano: valor alto (> 0.02).
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    count = sum(1 for w in words if w in _FIRST_PERSON)
    return count / len(words)


def compute_hedge_word_ratio(text: str) -> float:
    """
    Proporção de palavras de hedging (incerteza, opinião tentativa).

    Humanos naturalmente expressam incerteza ('maybe', 'I think', 'seems').
    LLMs são assertivos e confiantes ao extremo — raramente hesitam.

    Sinal de humano: valor alto (> 0.015).
    """
    text_lower = text.lower()
    words = _tokenize(text_lower)
    if not words:
        return 0.0

    count = 0
    for phrase in _HEDGE_WORDS:
        if " " in phrase:
            count += text_lower.count(phrase)
        else:
            count += words.count(phrase)

    return count / len(words)


def compute_question_density(text: str) -> float:
    """
    Proporção de perguntas em relação ao total de sentenças.

    Humanos fazem perguntas retóricas, pedem reflexão, se questionam.
    LLMs raramente usam interrogativas — preferem afirmações declarativas.

    Sinal de humano: valor > 0.05.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    question_marks = text.count("?")
    return question_marks / len(sentences)


def compute_bigram_repetition_score(text: str) -> float:
    """
    Taxa de bigrams repetidos em relação ao total de bigrams.

    LLMs repetem padrões fraseológicos (ex: 'it is important', 'the fact that',
    'in the context of'). A distribuição de bigrams de IA é menos diversa.

    Baseado no princípio de Zipf: textos humanos têm distribuição de frequência
    de palavras com cauda longa mais pronunciada.

    Sinal de IA: valor alto (> 0.15).
    """
    words = _tokenize(text)
    if len(words) < 4:
        return 0.0
    bgs = _bigrams(words)
    counts = Counter(bgs)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts) if counts else 0.0


def compute_lexical_diversity_entropy(text: str) -> float:
    """
    Entropia de Shannon da distribuição de frequência de palavras.

    H = -sum(p * log2(p))

    Textos humanos têm maior entropia (distribuição mais uniforme — Zipf menos
    estrito). LLMs convertem para um vocabulário mais previsível e repetitivo,
    resultando em entropia menor por token.

    Normalizado pelo log2(vocab_size) para ficar entre 0 e 1.
    Sinal de humano: valor alto (> 0.85).
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    vocab_size = len(counts)
    if vocab_size < 2:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(vocab_size)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_hapax_legomena_ratio(text: str) -> float:
    """
    Proporção de palavras que aparecem exatamente uma vez (hapax legomena).

    Lei de Zipf aplicada: em textos humanos naturais, ~40-60% das palavras
    únicas aparecem apenas uma vez. LLMs tendem a repetir mais suas palavras-chave,
    resultando em menos hapax.

    Sinal de humano: valor alto (> 0.50).
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(counts) if counts else 0.0


# ---------------------------------------------------------------------------
# Feature vector e nomes (ordem FIXA — deve ser igual em treino e inferência)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Grupo 1: Heurísticas originais
    "avg_sentence_length",       # 0
    "vocabulary_richness",       # 1
    "burstiness",                # 2
    "punctuation_density",       # 3
    "avg_word_length",           # 4
    # Grupo 2: Features avançadas
    "transition_word_density",   # 5
    "first_person_ratio",        # 6
    "hedge_word_ratio",          # 7
    "question_density",          # 8
    "bigram_repetition_score",   # 9
    "lexical_diversity_entropy", # 10
    "hapax_legomena_ratio",      # 11
]

N_FEATURES_V3 = 12
N_FEATURES_V1 = 5


def extract_features(text: str, n_features: int = N_FEATURES_V3) -> list[float]:
    """
    Extrai o vetor de features na ordem definida em FEATURE_NAMES.
    n_features=12 para modelo v3, n_features=5 para fallback v1.
    """
    f = [
        compute_avg_sentence_length(text),
        compute_vocabulary_richness(text),
        compute_burstiness(text),
        compute_punctuation_density(text),
        compute_avg_word_length(text),
        compute_transition_word_density(text),
        compute_first_person_ratio(text),
        compute_hedge_word_ratio(text),
        compute_question_density(text),
        compute_bigram_repetition_score(text),
        compute_lexical_diversity_entropy(text),
        compute_hapax_legomena_ratio(text),
    ]
    return [round(v, 6) for v in f[:n_features]]


# ---------------------------------------------------------------------------
# Pontuação heurística (regras manuais sobre as 12 features)
# ---------------------------------------------------------------------------

def _heuristic_score(features: list[float]) -> float:
    """
    Score baseado em thresholds empíricos para cada feature.
    Funciona sem nenhum modelo ML — puro conhecimento do domínio.
    """
    score = 0.0
    n = len(features)

    # Grupo 1
    avg_sl = features[0] if n > 0 else 0
    vocab = features[1] if n > 1 else 0
    burst = features[2] if n > 2 else 1
    punct = features[3] if n > 3 else 0
    avg_wl = features[4] if n > 4 else 0

    if 18.0 <= avg_sl <= 26.0:   score += 0.12
    if 0.35 <= vocab <= 0.55:    score += 0.10
    if burst < 0.40:             score += 0.12
    if 0.03 <= punct <= 0.07:    score += 0.08
    if avg_wl > 5.0:             score += 0.08

    # Grupo 2 (só se disponíveis)
    if n >= 12:
        trans  = features[5]
        fp     = features[6]
        hedge  = features[7]
        quest  = features[8]
        bigram = features[9]
        entr   = features[10]
        hapax  = features[11]

        if trans > 0.025:   score += 0.12   # alta densidade de transitivos → IA
        if fp < 0.010:      score += 0.10   # sem primeira pessoa → IA
        if hedge < 0.008:   score += 0.08   # sem hedging → IA
        if quest < 0.02:    score += 0.05   # sem perguntas → IA
        if bigram > 0.15:   score += 0.07   # bigrams repetidos → IA
        if entr < 0.80:     score += 0.05   # baixa entropia → IA
        if hapax < 0.45:    score += 0.03   # poucos hapax → IA

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
# Camada 2 — Random Forest (com auto-detecção de versão)
# ---------------------------------------------------------------------------

def _load_rf():
    """
    Carrega o melhor modelo RF disponível com cache em memória.
    Preferência: v3 (12 features) > HC3 (5 features).
    """
    global _rf_model, _rf_n_features
    if _rf_model is not None:
        return _rf_model

    for path, n_feat in [(_RF_V3_PATH, N_FEATURES_V3), (_RF_HC3_PATH, N_FEATURES_V1)]:
        if path.exists():
            try:
                import joblib
                _rf_model = joblib.load(path)
                _rf_n_features = n_feat
                logger.info("RF model loaded: %s (%d features)", path.name, n_feat)
                return _rf_model
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path.name, exc)

    logger.info("No RF model found — heuristics only")
    return None


def _rf_probability(features_12: list[float]) -> float | None:
    """Retorna probabilidade de IA do RF, ou None se não disponível."""
    model = _load_rf()
    if model is None:
        return None
    try:
        import numpy as np
        feat = features_12[:_rf_n_features]
        prob = model.predict_proba(np.array([feat]))[0][1]
        return float(prob)
    except Exception as exc:
        logger.warning("RF inference failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# API pública — legada (mantida para SubmissionService)
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> dict:
    """Análise heurística pura. Mantida por compatibilidade."""
    start = time.perf_counter()
    features = extract_features(text, n_features=N_FEATURES_V3)
    score = _heuristic_score(features)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "ai_probability_score": round(score, 4),
        "confidence_level": _confidence(score),
        "verdict": _verdict(score),
        "avg_sentence_length": round(features[0], 4),
        "vocabulary_richness": round(features[1], 4),
        "burstiness_score": round(features[2], 4),
        "punctuation_density": round(features[3], 4),
        "avg_word_length": round(features[4], 4),
        "model_version": MODEL_VERSION,
        "processing_time_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# API pública — Cascade (Camada 1 + 2, orquestrado pelo endpoint)
# ---------------------------------------------------------------------------

def analyze_with_cascade(text: str) -> dict:
    """
    Analisa texto combinando as 12 features + RF (se disponível).

    Retorna dict completo com todos os sinais para o endpoint decidir
    se aciona Claude (Camada 3).
    """
    start = time.perf_counter()

    features = extract_features(text, n_features=N_FEATURES_V3)
    h_score = _heuristic_score(features)
    rf_prob = _rf_probability(features)

    if rf_prob is not None:
        final_score = round(0.30 * h_score + 0.70 * rf_prob, 4)
        used_version = "cascade-rf-v3" if _rf_n_features == N_FEATURES_V3 else "cascade-rf-v1"
    else:
        final_score = round(h_score, 4)
        used_version = CASCADE_VERSION

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    needs_claude = abs(final_score - 0.5) < 0.15

    # Dict de features nomeadas para interpretabilidade
    named_features = {FEATURE_NAMES[i]: features[i] for i in range(len(features))}

    return {
        "ai_probability_score": final_score,
        "confidence_level": _confidence(final_score),
        "verdict": _verdict(final_score),
        "needs_claude": needs_claude,
        "heuristic_score": round(h_score, 4),
        "ml_score": round(rf_prob, 4) if rf_prob is not None else None,
        "features": features,
        "named_features": named_features,
        "model_version": used_version,
        "processing_time_ms": elapsed_ms,
    }
