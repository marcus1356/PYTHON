"""
Ativa o modelo v4 no detection_service.py.

Executar após o treinamento:
    python scripts/update_service_v4.py

O que faz:
  1. Verifica se detector_v4_best.joblib existe
  2. Adiciona as 8 novas funções compute_* ao detection_service.py
  3. Atualiza extract_features() para incluir as 20 features
  4. Atualiza o path do modelo para v4
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "app" / "models" / "ml"
SVC_PATH   = ROOT / "app" / "services" / "detection_service.py"


def main():
    # Verifica modelo
    v4_path = MODELS_DIR / "detector_v4_best.joblib"
    if not v4_path.exists():
        print("[ERRO] detector_v4_best.joblib não encontrado.")
        print("       Execute primeiro: python scripts/train_v4.py")
        sys.exit(1)

    # Verifica métricas
    metrics_path = MODELS_DIR / "v4_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        acc = m.get("test_accuracy", 0) * 100
        print(f"[INFO] Modelo v4 treinado — Accuracy: {acc:.2f}%")
        baseline = m.get("baseline_v3_acc")
        if baseline:
            improvement = m.get("improvement_vs_v3", 0) * 100
            print(f"       Baseline v3: {baseline*100:.2f}%  |  Melhora: {improvement:+.2f} pp")

    # Lê o serviço atual
    service_text = SVC_PATH.read_text(encoding="utf-8")

    # Verifica se já foi atualizado
    if "N_FEATURES_V4" in service_text:
        print("[INFO] detection_service.py já está atualizado para v4.")
        return

    # ── Bloco 1: Novas constantes e imports no topo ───────────────────────────
    NEW_IMPORTS = """
import re as _re_v4
try:
    from nltk.corpus import stopwords as _nltk_sw
    _SW_EN_V4 = set(_nltk_sw.words("english"))
except Exception:
    _SW_EN_V4 = set()

try:
    from nltk.tokenize import sent_tokenize as _sent_tokenize_v4
except Exception:
    import re as _re_sent
    def _sent_tokenize_v4(text):
        return [s.strip() for s in _re_sent.split(r"[.!?]+", text) if s.strip()]
"""

    # ── Bloco 2: 8 novas funções compute_* ───────────────────────────────────
    NEW_FEATURES_CODE = '''

# ---------------------------------------------------------------------------
# FEATURE GROUP 3: Features v4 (estilométricas / linguísticas — 8 novas)
# ---------------------------------------------------------------------------

def compute_readability_fog(text: str) -> float:
    """Proporção de palavras longas (>= 7 chars) como proxy do Gunning Fog."""
    words = re.findall(r"\\b[a-zA-Z]+\\b", text)
    if not words:
        return 0.0
    return sum(1 for w in words if len(w) >= 7) / len(words)


def compute_stopword_ratio(text: str) -> float:
    """Proporção de stopwords — humanos usam ligeiramente mais."""
    words = re.findall(r"\\b[a-zA-Z]+\\b", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in _SW_EN_V4) / len(words)


def compute_sentence_length_variance(text: str) -> float:
    """Variância normalizada do comprimento das frases. Humanos são mais irregulares."""
    import numpy as _np
    sentences = _sent_tokenize_v4(text)
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean_l  = sum(lengths) / len(lengths)
    if mean_l < 1:
        return 0.0
    var = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
    return float(var / (mean_l ** 2))


def compute_comma_density(text: str) -> float:
    """Média de vírgulas por frase normalizada em [0, 1]. IA cria listas repetidamente."""
    sentences = _sent_tokenize_v4(text)
    if not sentences:
        return 0.0
    avg = sum(s.count(",") for s in sentences) / len(sentences)
    return min(avg / 5.0, 1.0)


def compute_unique_trigrams_ratio(text: str) -> float:
    """Proporção de trigrams únicos. IA repete padrões → valor mais baixo."""
    words = re.findall(r"\\b[a-zA-Z]+\\b", text.lower())
    if len(words) < 3:
        return 0.0
    trigrams = list(zip(words, words[1:], words[2:]))
    return len(set(trigrams)) / len(trigrams)


def compute_pos_noun_ratio(text: str) -> float:
    """Proporção de substantivos (NN*). IA usa mais substantivos abstratos."""
    try:
        import nltk
        words = nltk.word_tokenize(text)[:300]
        if not words:
            return 0.0
        tags  = nltk.pos_tag(words)
        return sum(1 for _, t in tags if t.startswith("NN")) / len(words)
    except Exception:
        return 0.0


def compute_coherence_score(text: str) -> float:
    """Similaridade cosseno TF-IDF entre frases adjacentes. IA é mais coerente."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        sents = _sent_tokenize_v4(text)[:20]
        if len(sents) < 3:
            return 0.0
        vec   = TfidfVectorizer(max_features=200, stop_words="english")
        tfidf = vec.fit_transform(sents)
        sims  = [cosine_similarity(tfidf[i], tfidf[i + 1])[0][0] for i in range(len(sents) - 1)]
        return float(sum(sims) / len(sims))
    except Exception:
        return 0.0


def compute_exclamation_ratio(text: str) -> float:
    """Proporção de frases exclamativas. Humanos são mais expressivos."""
    sentences = _sent_tokenize_v4(text)
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if s.strip().endswith("!")) / len(sentences)

'''

    # ── Bloco 3: feature names v4 ─────────────────────────────────────────────
    NEW_FEATURE_NAMES = '''

# v4: 20 features (12 originais + 8 novas)
FEATURE_NAMES_V4 = FEATURE_NAMES + [
    "readability_fog",           # 12
    "stopword_ratio",            # 13
    "sentence_length_variance",  # 14
    "comma_density",             # 15
    "unique_trigrams_ratio",     # 16
    "pos_noun_ratio",            # 17
    "coherence_score",           # 18
    "exclamation_ratio",         # 19
]
N_FEATURES_V4 = len(FEATURE_NAMES_V4)
'''

    # ── Bloco 4: nova função extract_features_v4 ──────────────────────────────
    NEW_EXTRACT_FN = '''

def extract_features_v4(text: str) -> list[float]:
    """
    Extrai as 20 features do modelo v4.
    Compatível com CalibratedXGBoost salvo em detector_v4_best.joblib.
    """
    base = extract_features(text, n_features=N_FEATURES_V3)
    new_feats = [
        compute_readability_fog(text),
        compute_stopword_ratio(text),
        compute_sentence_length_variance(text),
        compute_comma_density(text),
        compute_unique_trigrams_ratio(text),
        compute_pos_noun_ratio(text),
        compute_coherence_score(text),
        compute_exclamation_ratio(text),
    ]
    return [round(v, 6) for v in base + new_feats]

'''

    # ── Aplica as alterações ──────────────────────────────────────────────────

    # 1. Adiciona imports após o bloco de imports existente
    service_text = service_text.replace(
        "logger = logging.getLogger(__name__)",
        "logger = logging.getLogger(__name__)" + NEW_IMPORTS,
        1,
    )

    # 2. Adiciona funções v4 antes de FEATURE_NAMES
    insert_marker = "FEATURE_NAMES = ["
    service_text = service_text.replace(
        insert_marker,
        NEW_FEATURES_CODE + insert_marker,
        1,
    )

    # 3. Adiciona FEATURE_NAMES_V4 e N_FEATURES_V4 após N_FEATURES_V1
    service_text = service_text.replace(
        "N_FEATURES_V1 = 5",
        "N_FEATURES_V1 = 5" + NEW_FEATURE_NAMES,
        1,
    )

    # 4. Adiciona extract_features_v4 após extract_features
    # Encontra o fim da função extract_features
    extract_end_marker = "    return [round(v, 6) for v in f[:n_features]]"
    service_text = service_text.replace(
        extract_end_marker,
        extract_end_marker + NEW_EXTRACT_FN,
        1,
    )

    # 5. Atualiza o path do modelo v4
    service_text = service_text.replace(
        '_RF_V3_PATH = _ML_DIR / "detector_v3_rf.joblib"',
        '_RF_V3_PATH = _ML_DIR / "detector_v3_rf.joblib"\n_RF_V4_PATH = _ML_DIR / "detector_v4_best.joblib"',
        1,
    )

    # 6. Atualiza a função de carga do modelo para preferir v4
    old_load = '''_rf_model = None
_rf_n_features: int = 0  # quantas features o modelo carregado espera'''
    new_load = '''_rf_model = None
_rf_n_features: int = 0  # quantas features o modelo carregado espera
_rf_uses_v4: bool = False'''
    service_text = service_text.replace(old_load, new_load, 1)

    SVC_PATH.write_text(service_text, encoding="utf-8")
    print("[OK] detection_service.py atualizado com suporte a v4.")

    # ── Atualiza _load_rf_model para carregar v4 ─────────────────────────────
    _patch_load_function()


def _patch_load_function():
    """Atualiza _load_rf_model() para preferir o modelo v4."""
    svc_text = SVC_PATH.read_text(encoding="utf-8")

    if "_RF_V4_PATH.exists()" in svc_text:
        print("[INFO] Função _load_rf_model já está atualizada para v4.")
        return

    # Encontra a função e substitui
    old_fn = '''def _load_rf_model() -> None:
    global _rf_model, _rf_n_features
    if _rf_model is not None:
        return'''

    new_fn = '''def _load_rf_model() -> None:
    global _rf_model, _rf_n_features, _rf_uses_v4
    if _rf_model is not None:
        return
    # Prefere o modelo v4 (20 features, XGBoost calibrado)
    if _RF_V4_PATH.exists():
        try:
            _rf_model      = joblib.load(_RF_V4_PATH)
            _rf_n_features = N_FEATURES_V4
            _rf_uses_v4    = True
            logger.info("Modelo v4 carregado: %s (20 features)", _RF_V4_PATH.name)
            return
        except Exception as exc:
            logger.warning("Falha ao carregar v4, voltando para v3: %s", exc)'''

    if old_fn in svc_text:
        svc_text = svc_text.replace(old_fn, new_fn, 1)
        SVC_PATH.write_text(svc_text, encoding="utf-8")
        print("[OK] _load_rf_model() atualizado para carregar v4.")
    else:
        print("[AVISO] Não foi possível localizar _load_rf_model(). Ajuste manual necessário.")

    # ── Atualiza analyze_with_cascade para usar extract_features_v4 ──────────
    svc_text = SVC_PATH.read_text(encoding="utf-8")
    old_extract = "features = extract_features(text, n_features=_rf_n_features)"
    new_extract = (
        "features = extract_features_v4(text) if _rf_uses_v4 "
        "else extract_features(text, n_features=_rf_n_features)"
    )
    if old_extract in svc_text:
        svc_text = svc_text.replace(old_extract, new_extract, 1)
        SVC_PATH.write_text(svc_text, encoding="utf-8")
        print("[OK] analyze_with_cascade() atualizado para usar features v4.")
    else:
        print("[AVISO] Não foi possível atualizar analyze_with_cascade(). Ajuste manual necessário.")


if __name__ == "__main__":
    main()
