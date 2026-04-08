"""
Calibração de thresholds por domínio textual.

Classifica os textos do checkpoint em domínios (academic / news / social / general)
usando features já extraídas e encontra o threshold de decisão ótimo por domínio
via maximização de F1 no conjunto de teste (mesma divisão do treinamento).

Saída:
    app/models/ml/v4_domain_thresholds.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


# Necessário para desserializar detector_v4_best.joblib (salvo com __main__.PlattCalibrated)
class PlattCalibrated:
    def __init__(self, estimator):
        self.estimator = estimator
        self._scaler   = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def fit(self, X, y):
        probs = self.estimator.predict_proba(X)[:, 1].reshape(-1, 1)
        self._scaler.fit(probs, y)
        return self

    def predict_proba(self, X):
        probs     = self.estimator.predict_proba(X)[:, 1].reshape(-1, 1)
        cal_probs = self._scaler.predict_proba(probs)[:, 1]
        return np.column_stack([1 - cal_probs, cal_probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "app" / "models" / "ml"
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("calibrate_domains")

# ===========================================================================
# Índices das features v4 usados na classificação de domínio
# ===========================================================================
# 0  avg_sentence_length
# 1  vocabulary_richness
# 6  first_person_ratio
# 7  hedge_word_ratio
# 12 readability_fog
# 14 sentence_length_variance
# 19 exclamation_ratio

def classify_domain(features: np.ndarray) -> str:
    """
    Classifica o domínio textual com base nas features v4.
    Compatível com features extraídas tanto do checkpoint quanto de textos novos.
    """
    avg_sl  = float(features[0])   # comprimento médio de sentenças
    vocab   = float(features[1])   # riqueza de vocabulário
    fp      = float(features[6])   # pronomes de 1a pessoa
    hedge   = float(features[7])   # palavras de hedging
    fog     = float(features[12])  # proporção de palavras longas
    sl_var  = float(features[14])  # variância no tamanho de sentenças
    excl    = float(features[19])  # proporção de frases exclamativas

    # Academic: sentenças longas, vocabulário rico, hedging, sem exclamação
    if avg_sl > 22 and vocab > 0.52 and hedge > 0.035 and excl < 0.02:
        return "academic"

    # Social: sentenças curtas, exclamativo ou muito 1a pessoa
    if avg_sl < 14 and (excl > 0.06 or fp > 0.12):
        return "social"

    # News: sentenças médias, quase sem 1a pessoa, sem exclamação
    if 14 <= avg_sl <= 23 and fp < 0.025 and excl < 0.015:
        return "news"

    return "general"


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Busca o threshold que maximiza F1 weighted em passos de 0.01."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.10, 0.90, 161):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return round(float(best_t), 3), round(float(best_f1), 4)


def main():
    ckpt_path  = MODELS_DIR / "v4_features_checkpoint.npz"
    model_path = MODELS_DIR / "detector_v4_best.joblib"
    out_path   = MODELS_DIR / "v4_domain_thresholds.json"

    if not ckpt_path.exists():
        log.error("Checkpoint nao encontrado: %s", ckpt_path)
        sys.exit(1)
    if not model_path.exists():
        log.error("Modelo nao encontrado: %s", model_path)
        sys.exit(1)

    log.info("Carregando checkpoint...")
    ckpt = np.load(ckpt_path)
    X    = ckpt["X"].astype(np.float32)
    y    = ckpt["y"]
    log.info("  Shape: X=%s  y=%s", X.shape, y.shape)

    # Reproduz exatamente o split do treinamento
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    log.info("Conjunto de teste: %d amostras", len(X_test))

    log.info("Carregando modelo v4...")
    import joblib
    model = joblib.load(model_path)

    log.info("Computando probabilidades no teste...")
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classificar domínios
    log.info("Classificando domínios...")
    domains_arr = np.array([classify_domain(X_test[i]) for i in range(len(X_test))])

    # Global baseline
    global_t, global_f1 = find_best_threshold(y_test, y_prob)
    global_auc = roc_auc_score(y_test, y_prob)
    log.info("Global -> threshold=%.3f  F1=%.4f  AUC=%.4f", global_t, global_f1, global_auc)

    thresholds = {
        "default": {
            "threshold":      global_t,
            "f1":             global_f1,
            "auc":            round(float(global_auc), 4),
            "n_samples":      int(len(y_test)),
        }
    }

    log.info("=" * 55)
    log.info("%-10s %8s %8s %8s %8s", "Dominio", "N", "Thresh", "F1", "AUC")
    log.info("-" * 55)

    for domain in ["academic", "news", "social", "general"]:
        mask = domains_arr == domain
        n    = int(mask.sum())
        if n < 500:
            log.info("  %-10s n=%d — amostras insuficientes, usando default", domain, n)
            continue

        y_d = y_test[mask]
        p_d = y_prob[mask]

        # Precisa de ambas as classes
        if len(np.unique(y_d)) < 2:
            log.info("  %-10s n=%d — apenas uma classe, usando default", domain, n)
            continue

        t, f1 = find_best_threshold(y_d, p_d)
        auc   = round(float(roc_auc_score(y_d, p_d)), 4)

        log.info("  %-10s %8d %8.3f %8.4f %8.4f", domain, n, t, f1, auc)
        thresholds[domain] = {
            "threshold": t,
            "f1":        f1,
            "auc":       auc,
            "n_samples": n,
        }

    log.info("=" * 55)

    # Distribuição de domínios
    unique, counts = np.unique(domains_arr, return_counts=True)
    log.info("Distribuicao de dominios no teste:")
    for d, c in zip(unique, counts):
        log.info("  %-10s %d (%.1f%%)", d, c, 100 * c / len(domains_arr))

    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("Thresholds salvos: %s", out_path)


if __name__ == "__main__":
    main()
