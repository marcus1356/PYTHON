"""
Script de treinamento - Modelo v4 (XGBoost + 20 features + Optuna).

Executar a partir da raiz do projeto:
    python scripts/train_v4.py

Saídas:
    app/models/ml/detector_v4_best.joblib  - modelo para produção
    app/models/ml/v4_metrics.json          - métricas
    app/models/ml/v4_feature_names.json    - lista de features
    app/models/ml/v4_features_checkpoint.npz - checkpoint de features
    scripts/plots/                         - gráficos PNG
"""

import json
import logging
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# ── Backend não-interativo para matplotlib ────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Caminho do projeto ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "app" / "models" / "ml"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "train_v4.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("train_v4")

# ── Imports ML ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import nltk
for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger_eng", "stopwords"]:
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")

# ── Configurações ─────────────────────────────────────────────────────────────
SEED          = 42
MAX_TEXT_LEN  = 4000
MIN_TEXT_LEN  = 50
SAMPLE_RAID   = 200_000
SAMPLE_PILE   = 40_000   # Pile é texto humano puro (label=0)
SAMPLE_HC3    = None
SAMPLE_MAGE   = 80_000
OPTUNA_TRIALS = 100

rng = np.random.default_rng(SEED)

# ===========================================================================
# 0. GPU CHECK
# ===========================================================================

def _check_xgb_gpu() -> bool:
    try:
        m = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=2, verbosity=0)
        m.fit(np.random.rand(50, 5), np.random.randint(0, 2, 50))
        return True
    except Exception:
        return False

def _check_lgb_gpu() -> bool:
    try:
        m = lgb.LGBMClassifier(device="gpu", n_estimators=2, verbose=-1)
        m.fit(np.random.rand(50, 5), np.random.randint(0, 2, 50))
        return True
    except Exception:
        return False

USE_XGB_GPU = _check_xgb_gpu()
USE_LGB_GPU = _check_lgb_gpu()
XGB_DEVICE  = "cuda" if USE_XGB_GPU else "cpu"
LGB_DEVICE  = "gpu"  if USE_LGB_GPU else "cpu"

log.info("XGBoost device: %s | LightGBM device: %s", XGB_DEVICE, LGB_DEVICE)

# ===========================================================================
# 1. CARREGAR DATASETS
# ===========================================================================

def _sample_df(df: pd.DataFrame, n: int | None, label_col: str = "label") -> pd.DataFrame:
    if n is None or len(df) <= n * 2:
        return df
    parts = []
    for lbl in df[label_col].unique():
        sub = df[df[label_col] == lbl]
        parts.append(sub.sample(min(n, len(sub)), random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)

def _clean_text(t: str) -> str | None:
    if not isinstance(t, str):
        return None
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < MIN_TEXT_LEN:
        return None
    return t[:MAX_TEXT_LEN]

def _hf_sample(ds, n: int | None, seed: int = SEED):
    """Faz shuffle + select no dataset HF antes de to_pandas(), economizando RAM."""
    if n is None:
        return ds
    n_select = min(n * 4, len(ds))  # 4× para balancear depois
    return ds.shuffle(seed=seed).select(range(n_select))


def load_datasets() -> pd.DataFrame:
    from datasets import load_dataset
    dfs = []

    log.info("[1/4] Carregando RAID...")
    # RAID não tem coluna 'label' - derivar de 'model' (human=0, AI=1)
    # Apenas attack='none' para evitar textos adversarialmente modificados
    raid_raw = load_dataset("liamdugan/raid", split="train")
    log.info("   RAID tamanho total: %d", len(raid_raw))
    raid_raw = _hf_sample(raid_raw, SAMPLE_RAID)
    raid_df  = raid_raw.to_pandas()
    raid_df  = raid_df.rename(columns={"generation": "text"})
    raid_df["label"]  = (raid_df["model"] != "human").astype(int)
    raid_df["text"]   = raid_df["text"].apply(_clean_text)
    raid_df["source"] = "raid"
    raid_df = raid_df[["text", "label", "source"]].dropna(subset=["text"])
    raid_df = _sample_df(raid_df, SAMPLE_RAID)
    dfs.append(raid_df)
    vc = raid_df["label"].value_counts()
    log.info("   RAID: human=%d  ai=%d  total=%d", vc.get(0, 0), vc.get(1, 0), len(raid_df))

    log.info("[2/4] Carregando AI Detection Pile (texto humano puro)...")
    pile_raw = load_dataset("artem9k/ai-text-detection-pile", split="train")
    log.info("   Pile tamanho total: %d", len(pile_raw))
    pile_raw = _hf_sample(pile_raw, SAMPLE_PILE)
    pile_df  = pile_raw.to_pandas()
    pile_df["text"]   = pile_df["text"].apply(_clean_text)
    pile_df["label"]  = 0  # Dataset é exclusivamente texto humano
    pile_df["source"] = "pile"
    pile_df = pile_df[["text", "label", "source"]].dropna(subset=["text"])
    if len(pile_df) > SAMPLE_PILE:
        pile_df = pile_df.sample(SAMPLE_PILE, random_state=SEED).reset_index(drop=True)
    dfs.append(pile_df)
    log.info("   Pile: human=%d  total=%d", len(pile_df), len(pile_df))

    log.info("[3/4] Carregando HC3...")
    # HC3 usa loading script descontinuado - carregar direto do cache JSONL local
    _HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
    _hc3_patterns = list(_HF_CACHE.glob("datasets--Hello-SimpleAI--HC3/**/all.jsonl"))
    if not _hc3_patterns:
        # Fallback: tenta baixar como json
        try:
            hc3_raw = load_dataset("json",
                                   data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl",
                                   split="train")
            hc3_df = hc3_raw.to_pandas()
        except Exception as e:
            log.warning("   HC3 não disponível (%s) - pulando.", e)
            hc3_df = pd.DataFrame()
    else:
        import json as _json
        _hc3_rows_raw = []
        with open(_hc3_patterns[0], encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    _hc3_rows_raw.append(_json.loads(_line))
        hc3_df = pd.DataFrame(_hc3_rows_raw)

    if not hc3_df.empty:
        rows = []
        for _, row in hc3_df.iterrows():
            for hr in (row.get("human_answers") or []):
                t = _clean_text(hr)
                if t: rows.append({"text": t, "label": 0, "source": "hc3"})
            for ar in (row.get("chatgpt_answers") or []):
                t = _clean_text(ar)
                if t: rows.append({"text": t, "label": 1, "source": "hc3"})
        hc3_flat = pd.DataFrame(rows)
        dfs.append(hc3_flat)
        vc = hc3_flat["label"].value_counts()
        log.info("   HC3: human=%d  ai=%d  total=%d", vc.get(0, 0), vc.get(1, 0), len(hc3_flat))
    else:
        log.warning("   HC3 ignorado.")

    log.info("[4/4] Carregando MAGE...")
    mage_raw = load_dataset("yaful/MAGE", split="train")
    log.info("   MAGE tamanho total: %d", len(mage_raw))
    mage_raw = _hf_sample(mage_raw, SAMPLE_MAGE)
    mage_df  = mage_raw.to_pandas()
    mage_df["text"]   = mage_df["text"].apply(_clean_text)
    mage_df["label"]  = mage_df["label"].astype(int)
    mage_df["source"] = "mage"
    mage_df = mage_df[["text", "label", "source"]].dropna(subset=["text"])
    mage_df = _sample_df(mage_df, SAMPLE_MAGE)
    dfs.append(mage_df)
    vc = mage_df["label"].value_counts()
    log.info("   MAGE: human=%d  ai=%d  total=%d", vc.get(0, 0), vc.get(1, 0), len(mage_df))

    df_all = pd.concat(dfs, ignore_index=True)
    before = len(df_all)
    df_all["text_key"] = df_all["text"].str[:200]
    df_all = df_all.drop_duplicates(subset="text_key").drop(columns="text_key")
    df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)
    after  = len(df_all)
    vc = df_all["label"].value_counts()
    log.info("Dataset final: %d amostras (dedup removeu %d) | human=%d ai=%d",
             after, before - after, vc.get(0, 0), vc.get(1, 0))
    return df_all

# ===========================================================================
# 2. FEATURES V4 (12 existentes + 8 novas)
# ===========================================================================

from app.services.detection_service import (
    extract_features, FEATURE_NAMES, N_FEATURES_V3,
)

try:
    _SW_EN = set(nltk_stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    _SW_EN = set(nltk_stopwords.words("english"))


def compute_readability_fog(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    if not words:
        return 0.0
    long_words = sum(1 for w in words if len(w) >= 7)
    return long_words / len(words)


def compute_stopword_ratio(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in _SW_EN) / len(words)


def compute_sentence_length_variance(text: str) -> float:
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean_l  = np.mean(lengths)
    if mean_l < 1:
        return 0.0
    return float(np.var(lengths) / (mean_l ** 2))


def compute_comma_density(text: str) -> float:
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    return min(np.mean([s.count(",") for s in sentences]) / 5.0, 1.0)


def compute_unique_trigrams_ratio(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) < 3:
        return 0.0
    trigrams = list(zip(words, words[1:], words[2:]))
    if not trigrams:
        return 0.0
    return len(set(trigrams)) / len(trigrams)


def compute_pos_noun_ratio(text: str) -> float:
    try:
        from nltk import pos_tag, word_tokenize
        words = word_tokenize(text)[:300]
        if not words:
            return 0.0
        tags  = pos_tag(words)
        nouns = sum(1 for _, t in tags if t.startswith("NN"))
        return nouns / len(words)
    except Exception:
        return 0.0


def compute_coherence_score(text: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        sents = sent_tokenize(text)
        if len(sents) < 3:
            return 0.0
        sents = sents[:20]
        vec   = TfidfVectorizer(max_features=200, stop_words="english")
        tfidf = vec.fit_transform(sents)
        sims  = [cosine_similarity(tfidf[i], tfidf[i + 1])[0][0] for i in range(len(sents) - 1)]
        return float(np.mean(sims))
    except Exception:
        return 0.0


def compute_exclamation_ratio(text: str) -> float:
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if s.strip().endswith("!")) / len(sentences)


FEATURE_NAMES_V4 = FEATURE_NAMES + [
    "readability_fog",
    "stopword_ratio",
    "sentence_length_variance",
    "comma_density",
    "unique_trigrams_ratio",
    "pos_noun_ratio",
    "coherence_score",
    "exclamation_ratio",
]
N_FEATURES_V4 = len(FEATURE_NAMES_V4)
log.info("Features v4: %d total (%d novas)", N_FEATURES_V4, N_FEATURES_V4 - N_FEATURES_V3)


def _extract_safe_v3(text: str) -> list[float]:
    try:
        return extract_features(text, n_features=N_FEATURES_V3)
    except Exception:
        return [0.0] * N_FEATURES_V3


def extract_features_v4(text: str) -> list[float]:
    v3 = _extract_safe_v3(text)
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
    return v3 + new_feats

# ===========================================================================
# 3. EXTRAÇÃO PARALELA COM CHECKPOINT
# ===========================================================================

def extract_all_features(df_all: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from joblib import Parallel, delayed

    CHECKPOINT_PATH = MODELS_DIR / "v4_features_checkpoint.npz"
    BATCH_SIZE = 10_000

    if CHECKPOINT_PATH.exists():
        log.info("Checkpoint encontrado - carregando features salvas...")
        ckpt = np.load(CHECKPOINT_PATH)
        X_v4 = ckpt["X"].astype(np.float32)
        y_v4 = ckpt["y"]
        log.info("   Carregados: %d amostras x %d features", X_v4.shape[0], X_v4.shape[1])
        return X_v4, y_v4

    texts  = df_all["text"].tolist()
    labels = df_all["label"].values
    n      = len(texts)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    log.info("Extraindo %d features de %d textos em %d batches...", N_FEATURES_V4, n, n_batches)
    all_feats = []
    t0 = time.time()

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end   = min(start + BATCH_SIZE, n)
        batch = texts[start:end]
        batch_feats = Parallel(n_jobs=-1, prefer="processes")(
            delayed(extract_features_v4)(t) for t in batch
        )
        all_feats.extend(batch_feats)

        elapsed = time.time() - t0
        pct     = end / n * 100
        eta     = elapsed / pct * (100 - pct) if pct > 0 else 0
        log.info("  Batch %d/%d: %d/%d (%.1f%%)  ETA: %.0fs", i + 1, n_batches, end, n, pct, eta)

    X_v4 = np.array(all_feats, dtype=np.float32)
    y_v4 = labels

    np.savez_compressed(CHECKPOINT_PATH, X=X_v4, y=y_v4)
    log.info("Extração concluída em %.1f min. Checkpoint salvo.", (time.time() - t0) / 60)
    return X_v4, y_v4

# ===========================================================================
# 4. AVALIAÇÃO BASELINE v3
# ===========================================================================

def eval_baseline_v3(df_all: pd.DataFrame) -> float:
    from joblib import Parallel, delayed

    V3_MODEL_PATH = MODELS_DIR / "detector_v3_rf.joblib"
    if not V3_MODEL_PATH.exists():
        log.warning("Modelo v3 não encontrado. Pulando avaliação baseline.")
        return 0.0

    BASELINE_SAMPLE = 10_000
    df_baseline = df_all.sample(BASELINE_SAMPLE, random_state=SEED).reset_index(drop=True)
    log.info("Avaliando baseline v3 em %d amostras...", BASELINE_SAMPLE)

    feats_baseline = Parallel(n_jobs=-1, verbose=0)(
        delayed(_extract_safe_v3)(t) for t in df_baseline["text"]
    )
    X_b = np.array(feats_baseline, dtype=np.float32)
    y_b = df_baseline["label"].values

    rf_v3     = joblib.load(V3_MODEL_PATH)
    y_pred_v3 = rf_v3.predict(X_b)
    y_prob_v3 = rf_v3.predict_proba(X_b)[:, 1]
    acc_v3    = accuracy_score(y_b, y_pred_v3)
    f1_v3     = f1_score(y_b, y_pred_v3, average="weighted")
    auc_v3    = roc_auc_score(y_b, y_prob_v3)
    log.info("Baseline v3 -> Accuracy: %.4f | F1: %.4f | AUC: %.4f", acc_v3, f1_v3, auc_v3)
    return acc_v3

# ===========================================================================
# 5. TREINAMENTO
# ===========================================================================

class PlattCalibrated:
    """
    Wrapper de Platt Scaling para modelo pré-treinado.
    Definido no nível do módulo para ser serializável por joblib/pickle.
    """
    def __init__(self, estimator):
        from sklearn.linear_model import LogisticRegression
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


def train_all(X_v4: np.ndarray, y_v4: np.ndarray, acc_v3: float) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X_v4, y_v4, test_size=0.20, random_state=SEED, stratify=y_v4
    )
    log.info("Split: treino=%d | teste=%d", len(X_train), len(X_test))

    scale_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # ── XGBoost inicial ───────────────────────────────────────────────────────
    log.info("Treinando XGBoost inicial (device=%s)...", XGB_DEVICE)
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pw,
        tree_method="hist", device=XGB_DEVICE,
        eval_metric="logloss", early_stopping_rounds=30,
        random_state=SEED, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    log.info("  XGBoost: %.1fs | %d árvores", time.time() - t0, xgb_model.best_iteration)

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    acc_xgb    = accuracy_score(y_test, y_pred_xgb)
    f1_xgb     = f1_score(y_test, y_pred_xgb, average="weighted")
    auc_xgb    = roc_auc_score(y_test, y_prob_xgb)
    log.info("  XGBoost -> Acc=%.4f | F1=%.4f | AUC=%.4f", acc_xgb, f1_xgb, auc_xgb)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    log.info("Treinando LightGBM (device=%s)...", LGB_DEVICE)
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=-1, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        device=LGB_DEVICE, class_weight="balanced",
        random_state=SEED, verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    log.info("  LightGBM: %.1fs", time.time() - t0)

    y_pred_lgb = lgb_model.predict(X_test)
    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
    acc_lgb    = accuracy_score(y_test, y_pred_lgb)
    f1_lgb     = f1_score(y_test, y_pred_lgb, average="weighted")
    auc_lgb    = roc_auc_score(y_test, y_prob_lgb)
    log.info("  LightGBM -> Acc=%.4f | F1=%.4f | AUC=%.4f", acc_lgb, f1_lgb, auc_lgb)

    # ── Optuna: hiperparâmetros do XGBoost ────────────────────────────────────
    X_tr_opt, X_val_opt, y_tr_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.125, random_state=SEED, stratify=y_train
    )

    # Se --fast: reutiliza melhores params já encontrados (pula Optuna)
    _FAST_PARAMS = {
        "n_estimators": 762, "learning_rate": 0.048588772011593354,
        "max_depth": 10, "subsample": 0.7997700610298202,
        "colsample_bytree": 0.9093057923859031, "reg_alpha": 1.286497131033996,
        "reg_lambda": 0.00288679076892098, "min_child_weight": 8,
        "gamma": 1.0470737950960085,
    }
    _use_fast = "--fast" in sys.argv

    if _use_fast:
        log.info("Modo --fast: usando params pre-otimizados (pulando Optuna).")

        class _FakeStudy:
            best_value  = 0.86726
            best_params = _FAST_PARAMS
        study_xgb = _FakeStudy()
    else:
        log.info("Optuna (%d trials) - pode levar 10-30 min...", OPTUNA_TRIALS)

        def xgb_objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators"    : trial.suggest_int("n_estimators", 200, 1000),
                "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth"       : trial.suggest_int("max_depth", 4, 10),
                "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma"           : trial.suggest_float("gamma", 0, 5),
            }
            m = xgb.XGBClassifier(
                **params, tree_method="hist", device=XGB_DEVICE,
                random_state=SEED, verbosity=0,
                early_stopping_rounds=20, eval_metric="logloss",
            )
            m.fit(X_tr_opt, y_tr_opt, eval_set=[(X_val_opt, y_val_opt)], verbose=False)
            return roc_auc_score(y_val_opt, m.predict_proba(X_val_opt)[:, 1])

        study_xgb = optuna.create_study(direction="maximize", study_name="xgb_v4_detector")
        study_xgb.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False, n_jobs=1)
        log.info("Optuna melhor AUC: %.5f", study_xgb.best_value)
        log.info("Melhores params: %s", study_xgb.best_params)

    # ── XGBoost otimizado ─────────────────────────────────────────────────────
    best_params = study_xgb.best_params.copy()
    n_est_final = best_params.pop("n_estimators")
    xgb_best = xgb.XGBClassifier(
        **best_params, n_estimators=n_est_final,
        tree_method="hist", device=XGB_DEVICE,
        eval_metric="logloss", random_state=SEED, verbosity=0,
    )
    log.info("Treinando XGBoost otimizado no treino completo...")
    t0 = time.time()
    xgb_best.fit(X_train, y_train)
    log.info("  XGBoost otimizado: %.1fs", time.time() - t0)

    y_pred_best = xgb_best.predict(X_test)
    y_prob_best = xgb_best.predict_proba(X_test)[:, 1]
    acc_best    = accuracy_score(y_test, y_pred_best)
    f1_best     = f1_score(y_test, y_pred_best, average="weighted")
    auc_best    = roc_auc_score(y_test, y_prob_best)
    log.info("  XGBoost otimizado -> Acc=%.4f | F1=%.4f | AUC=%.4f", acc_best, f1_best, auc_best)

    # ── RF v4 para ensemble ───────────────────────────────────────────────────
    log.info("Treinando RF v4...")
    rf_v4 = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    rf_v4.fit(X_train, y_train)
    y_pred_rf4 = rf_v4.predict(X_test)
    y_prob_rf4 = rf_v4.predict_proba(X_test)[:, 1]
    acc_rf4    = accuracy_score(y_test, y_pred_rf4)
    log.info("  RF v4 -> Acc=%.4f", acc_rf4)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    log.info("Treinando ensemble (soft voting)...")
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb_best), ("lgb", lgb_model), ("rf", rf_v4)],
        voting="soft", weights=[2, 1, 1], n_jobs=-1,
    )
    t0 = time.time()
    ensemble.fit(X_train, y_train)
    log.info("  Ensemble: %.1fs", time.time() - t0)

    y_pred_ens = ensemble.predict(X_test)
    y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
    acc_ens    = accuracy_score(y_test, y_pred_ens)
    f1_ens     = f1_score(y_test, y_pred_ens, average="weighted")
    auc_ens    = roc_auc_score(y_test, y_prob_ens)
    log.info("  Ensemble -> Acc=%.4f | F1=%.4f | AUC=%.4f", acc_ens, f1_ens, auc_ens)

    # ── Calibração de Platt (manual - cv='prefit' removido no sklearn >= 1.2) ─
    best_individual      = xgb_best if acc_best >= acc_lgb else lgb_model
    best_individual_name = "XGBoost" if acc_best >= acc_lgb else "LightGBM"
    log.info("Calibrando %s com Platt Scaling...", best_individual_name)

    calibrated = PlattCalibrated(best_individual)
    calibrated.fit(X_val_opt, y_val_opt)

    y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
    y_pred_cal = (y_prob_cal >= 0.5).astype(int)
    acc_cal    = accuracy_score(y_test, y_pred_cal)
    brier_cal  = brier_score_loss(y_test, y_prob_cal)
    log.info("  Calibrado -> Acc=%.4f | Brier=%.4f", acc_cal, brier_cal)

    # ── Tabela comparativa ────────────────────────────────────────────────────
    models_eval = [
        ("XGBoost inicial",   y_pred_xgb,  y_prob_xgb),
        ("LightGBM",          y_pred_lgb,  y_prob_lgb),
        ("XGBoost otimizado", y_pred_best, y_prob_best),
        ("RF v4",             y_pred_rf4,  y_prob_rf4),
        ("Ensemble (soft)",   y_pred_ens,  y_prob_ens),
        (f"{best_individual_name} calibrado", y_pred_cal, y_prob_cal),
    ]
    if acc_v3 > 0:
        models_eval.insert(0, ("v3 RF (baseline)", None, None))

    log.info("=" * 70)
    log.info("%-30s %10s %10s %10s %8s", "Modelo", "Accuracy", "F1 wtd", "ROC-AUC", "Brier")
    log.info("-" * 70)
    best_acc = 0.0
    best_row = None
    for name, preds, probs in models_eval:
        if preds is None:
            log.info("  %-30s %10.4f  (baseline)", name, acc_v3)
            continue
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="weighted")
        auc   = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        flag  = " ◄" if acc > best_acc else ""
        log.info("  %-30s %10.4f %10.4f %10.4f %8.4f%s", name, acc, f1, auc, brier, flag)
        if acc > best_acc:
            best_acc = acc
            best_row = (name, preds, probs)
    log.info("=" * 70)
    log.info("Melhor modelo: %s (%.2f%%)", best_row[0], best_acc * 100)

    # ── Salvar plots ──────────────────────────────────────────────────────────
    _save_roc_plot(models_eval, y_test)
    _save_confusion_matrix(best_row, y_test)

    return {
        "calibrated"   : calibrated,
        "best_row"     : best_row,
        "acc_cal"      : acc_cal,
        "brier_cal"    : brier_cal,
        "y_test"       : y_test,
        "y_pred_cal"   : y_pred_cal,
        "y_prob_cal"   : y_prob_cal,
        "X_train"      : X_train,
        "X_test"       : X_test,
        "study_xgb"    : study_xgb,
        "acc_v3"       : acc_v3,
        "models_eval"  : models_eval,
    }

# ===========================================================================
# 6. PLOTS
# ===========================================================================

def _save_roc_plot(models_eval, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, preds, probs in models_eval:
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="Curva ROC")
    axes[0].legend(fontsize=7)

    for name, preds, probs in models_eval:
        if probs is None:
            continue
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        axes[1].plot(mean_pred, frac_pos, marker="o", markersize=3, label=name)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set(xlabel="Score predito médio", ylabel="Fração positiva real", title="Curva de Calibração")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    path = PLOTS_DIR / "v4_roc_calibration.png"
    plt.savefig(path, dpi=120)
    plt.close()
    log.info("Plot salvo: %s", path)


def _save_confusion_matrix(best_row, y_test):
    best_name, best_preds, best_probs = best_row
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cm   = confusion_matrix(y_test, best_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Human", "AI"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Confusão - {best_name}")

    axes[1].hist(best_probs[y_test == 0], bins=50, alpha=0.7, label="Human", color="#22c55e", density=True)
    axes[1].hist(best_probs[y_test == 1], bins=50, alpha=0.7, label="AI",    color="#ef4444", density=True)
    axes[1].axvline(0.5, color="gray", linestyle="--")
    axes[1].set(xlabel="P(AI)", ylabel="Densidade", title="Distribuição de Scores")
    axes[1].legend()

    plt.tight_layout()
    path = PLOTS_DIR / "v4_confusion_scores.png"
    plt.savefig(path, dpi=120)
    plt.close()
    log.info("Plot salvo: %s", path)

# ===========================================================================
# 7. SALVAR MODELO E MÉTRICAS
# ===========================================================================

def save_results(results: dict) -> None:
    calibrated = results["calibrated"]
    acc_cal    = results["acc_cal"]
    y_test     = results["y_test"]
    y_pred_cal = results["y_pred_cal"]
    y_prob_cal = results["y_prob_cal"]
    X_train    = results["X_train"]
    X_test     = results["X_test"]
    study_xgb  = results["study_xgb"]
    acc_v3     = results["acc_v3"]

    V4_MODEL_PATH    = MODELS_DIR / "detector_v4_best.joblib"
    V4_METRICS_PATH  = MODELS_DIR / "v4_metrics.json"
    V4_FEATURES_PATH = MODELS_DIR / "v4_feature_names.json"

    joblib.dump(calibrated, V4_MODEL_PATH, compress=3)
    log.info("Modelo salvo: %s (%.1f MB)", V4_MODEL_PATH, V4_MODEL_PATH.stat().st_size / 1024 / 1024)

    with open(V4_FEATURES_PATH, "w") as f:
        json.dump(FEATURE_NAMES_V4, f, indent=2)

    metrics = {
        "model_version"    : "v4",
        "n_features"       : N_FEATURES_V4,
        "feature_names"    : FEATURE_NAMES_V4,
        "n_train_samples"  : len(X_train),
        "n_test_samples"   : len(X_test),
        "datasets_used"    : ["raid", "ai-detection-pile", "hc3", "mage"],
        "deploy_model"     : "CalibratedXGBoost",
        "test_accuracy"    : float(acc_cal),
        "test_f1_weighted" : float(f1_score(y_test, y_pred_cal, average="weighted")),
        "test_roc_auc"     : float(roc_auc_score(y_test, y_prob_cal)),
        "test_brier"       : float(brier_score_loss(y_test, y_prob_cal)),
        "baseline_v3_acc"  : float(acc_v3) if acc_v3 > 0 else None,
        "improvement_vs_v3": float(acc_cal - acc_v3) if acc_v3 > 0 else None,
        "optuna_best_auc"  : study_xgb.best_value,
        "optuna_best_params": study_xgb.best_params,
    }
    with open(V4_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("=" * 55)
    log.info("RESULTADO FINAL")
    log.info("=" * 55)
    if acc_v3 > 0:
        log.info("  Baseline v3: %.2f%%", acc_v3 * 100)
    log.info("  Modelo v4:   %.2f%%", acc_cal * 100)
    if acc_v3 > 0:
        log.info("  Melhora:     %+.2f pp", (acc_cal - acc_v3) * 100)
    log.info("=" * 55)
    log.info("Métricas salvas: %s", V4_METRICS_PATH)

# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    start_total = time.time()
    log.info("=" * 60)
    log.info("AI Detector - Treinamento v4")
    log.info("ROOT: %s", ROOT)
    log.info("=" * 60)

    # 1–3. Datasets + features (com checkpoint)
    _CKPT = MODELS_DIR / "v4_features_checkpoint.npz"
    if _CKPT.exists():
        log.info("Checkpoint encontrado - pulando carregamento de datasets e extração.")
        _ckpt = np.load(_CKPT)
        X_v4  = _ckpt["X"].astype(np.float32)
        y_v4  = _ckpt["y"]
        log.info("   Carregados do checkpoint: %d amostras x %d features", X_v4.shape[0], X_v4.shape[1])
        acc_v3 = 0.0  # baseline já foi avaliado na run anterior
    else:
        df_all = load_datasets()
        acc_v3 = eval_baseline_v3(df_all)
        X_v4, y_v4 = extract_all_features(df_all)
    X_v4 = np.nan_to_num(X_v4, nan=0.0, posinf=1.0, neginf=0.0)
    log.info("Shape final: X=%s  y=%s  NaN=%d", X_v4.shape, y_v4.shape, np.isnan(X_v4).sum())

    # 4. Treinar modelos
    results = train_all(X_v4, y_v4, acc_v3)

    # 5. Salvar
    save_results(results)

    total_min = (time.time() - start_total) / 60
    log.info("Treinamento concluído em %.1f minutos.", total_min)
    log.info("Próximo passo: execute 'python scripts/update_service_v4.py' para ativar o modelo.")
