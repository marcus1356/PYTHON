"""
Script de retrain — Detector v3 com 12 features.

Carrega datasets públicos do HuggingFace, extrai 12 features por texto,
treina Random Forest com cross-validation e salva o modelo.

Uso:
    python scripts/retrain_v3.py
"""

import json
import sys
import time
from pathlib import Path

# Garante que o root do projeto está no path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from app.services.detection_service import extract_features, FEATURE_NAMES, N_FEATURES_V3

MODEL_OUT = ROOT / "app" / "models" / "ml" / "detector_v3_rf.joblib"
RESULTS_OUT = ROOT / "app" / "models" / "ml" / "v3_metrics.json"


# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------

def load_datasets(max_per_class: int = 4000) -> tuple[list[str], list[int]]:
    """
    Tenta carregar datasets do HuggingFace.
    Retorna (texts, labels) com labels 0=humano, 1=IA.
    """
    texts: list[str] = []
    labels: list[int] = []

    print("\n[1/3] Carregando datasets...")

    # Dataset 1: NicolaiSivesind/ChatGPT-Research-Abstracts
    try:
        from datasets import load_dataset
        print("  -> NicolaiSivesind/ChatGPT-Research-Abstracts...")
        ds1 = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts", split="train")
        cols = ds1.column_names
        human_col = next((c for c in cols if "real" in c or "human" in c), None)
        ai_col = next((c for c in cols if "generated" in c or "ai" in c), None)
        if human_col and ai_col:
            n = min(max_per_class, len(ds1))
            for row in ds1.select(range(n)):
                h, a = row.get(human_col, ""), row.get(ai_col, "")
                if h and len(h) >= 100:
                    texts.append(h); labels.append(0)
                if a and len(a) >= 100:
                    texts.append(a); labels.append(1)
            print(f"     OK: {n} abstracts (x2 = {n*2} exemplos)")
        else:
            print(f"     AVISO: colunas inesperadas: {cols}")
    except Exception as e:
        print(f"     ERRO: {e}")

    # Dataset 2: dmitva/human_ai_generated_text
    try:
        from datasets import load_dataset
        print("  -> dmitva/human_ai_generated_text...")
        ds2 = load_dataset("dmitva/human_ai_generated_text", split="train")
        n = min(max_per_class, len(ds2))
        added = 0
        for row in ds2.select(range(n)):
            h = row.get("human_text", "")
            a = row.get("ai_text", "")
            if h and len(h) >= 100:
                texts.append(h); labels.append(0); added += 1
            if a and len(a) >= 100:
                texts.append(a); labels.append(1); added += 1
        print(f"     OK: {added} exemplos")
    except Exception as e:
        print(f"     ERRO: {e}")

    if not texts:
        print("\n  AVISO: Nenhum dataset carregado. Gerando dados sintéticos mínimos...")
        texts, labels = _synthetic_fallback()

    # Balanceia classes
    from collections import Counter
    c = Counter(labels)
    print(f"\n  Total: {len(texts)} exemplos | Humano: {c[0]} | IA: {c[1]}")

    return texts, labels


def _synthetic_fallback() -> tuple[list[str], list[int]]:
    """Dados sintéticos mínimos se HuggingFace não estiver disponível."""
    human = [
        "I was walking down the street when it started raining. Grabbed my jacket and ran. Totally soaked anyway. That's life, I guess? Makes you laugh afterwards.",
        "So I tried making pasta from scratch yesterday. Big mistake honestly. The dough kept tearing and I have no idea why. Maybe too much flour? My hands were a mess.",
        "Had this weird dream where I forgot my own name. Woke up panicking. Isn't it strange how your brain can just... blank like that? Still unsettled thinking about it.",
    ] * 30

    ai = [
        "Artificial intelligence has fundamentally transformed the landscape of modern technology. Furthermore, the systematic application of machine learning algorithms enables unprecedented analytical capabilities. Moreover, the integration of neural network architectures facilitates sophisticated pattern recognition.",
        "The implementation of sustainable practices is essential for environmental preservation. Additionally, it is important to note that renewable energy sources provide significant advantages. Consequently, organizations should prioritize the adoption of green technologies.",
        "In conclusion, effective communication is a cornerstone of successful leadership. It is worth noting that interpersonal skills play a crucial role in organizational dynamics. Therefore, investment in communication training yields substantial returns.",
    ] * 30

    texts = human + ai
    labels = [0] * len(human) + [1] * len(ai)
    return texts, labels


# ---------------------------------------------------------------------------
# Extração de features
# ---------------------------------------------------------------------------

def build_feature_matrix(
    texts: list[str], labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n[2/3] Extraindo {N_FEATURES_V3} features para {len(texts)} textos...")
    start = time.time()

    X = []
    y = []
    skipped = 0

    for i, (text, label) in enumerate(zip(texts, labels)):
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start
            rate = i / elapsed
            remaining = (len(texts) - i) / rate
            print(f"  {i}/{len(texts)} | {elapsed:.0f}s decorridos | ~{remaining:.0f}s restantes")
        try:
            feat = extract_features(text, n_features=N_FEATURES_V3)
            X.append(feat)
            y.append(label)
        except Exception:
            skipped += 1

    if skipped:
        print(f"  AVISO: {skipped} textos ignorados por erro")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    elapsed = time.time() - start
    print(f"  Concluido em {elapsed:.1f}s | Shape: {X_arr.shape}")
    return X_arr, y_arr


# ---------------------------------------------------------------------------
# Treino e avaliação
# ---------------------------------------------------------------------------

def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> tuple[object, dict]:
    print(f"\n[3/3] Treinando Random Forest com {len(X)} exemplos e {X.shape[1]} features...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # StratifiedKFold 5-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"\n  Cross-validation (5-fold):")
    for i, s in enumerate(cv_scores, 1):
        print(f"    Fold {i}: {s:.4f}")
    print(f"  Media: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Treina no dataset completo
    model.fit(X, y)
    y_pred = model.predict(X)

    print("\n  Classification report (treino completo):")
    report = classification_report(y, y_pred, target_names=["Humano", "IA"])
    print(report)

    # Feature importances
    importances = dict(zip(FEATURE_NAMES[:N_FEATURES_V3], model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
    print("\n  Feature importances:")
    for name, imp in sorted_imp:
        bar = "=" * int(imp * 50)
        print(f"    {name:<30} {imp:.4f}  {bar}")

    metrics = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_folds": cv_scores.tolist(),
        "n_train_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_importances": {k: float(v) for k, v in importances.items()},
        "model_class": "RandomForestClassifier",
        "n_estimators": 200,
    }

    return model, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("AI Detector — Retrain v3 (12 features)")
    print("=" * 60)

    texts, labels = load_datasets(max_per_class=4000)
    X, y = build_feature_matrix(texts, labels)
    model, metrics = train_and_evaluate(X, y)

    # Salva modelo
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModelo salvo em: {MODEL_OUT}")

    # Salva métricas
    with open(RESULTS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metricas salvas em: {RESULTS_OUT}")

    print(f"\nAcuracia final (CV): {metrics['cv_accuracy_mean']:.1%}")
    print("\nPara usar o novo modelo, reinicie o servidor FastAPI.")


if __name__ == "__main__":
    main()
