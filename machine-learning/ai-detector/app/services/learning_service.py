"""
Serviço de aprendizado contínuo (online learning).

Arquitetura:
  - SGDClassifier com partial_fit: aprende instantaneamente a cada novo exemplo
    rotulado (via /feedback). Persistido em disco com joblib.
  - Random Forest: retrain completo em background a cada RETRAIN_THRESHOLD novos
    exemplos confirmados. Substitui o modelo HC3 anterior.

Por que dois modelos?
  - SGD (online): latência O(1), atualiza em tempo real, mas menos preciso.
  - RF (batch): mais preciso, mas requer dados suficientes e tempo de treino.
  O endpoint /detect usa o RF para inferência; o SGD serve de "hot patch"
  entre retrains do RF.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ML_DIR = Path(__file__).resolve().parent.parent / "models" / "ml"
_SGD_PATH = _ML_DIR / "detector_sgd_online.joblib"
_RF_PATH = _ML_DIR / "detector_hc3_rf.joblib"

# Cache em memória
_sgd_model = None


def _load_sgd():
    """Carrega ou cria o SGDClassifier com cache em memória."""
    global _sgd_model
    if _sgd_model is not None:
        return _sgd_model

    if _SGD_PATH.exists():
        try:
            import joblib
            _sgd_model = joblib.load(_SGD_PATH)
            logger.info("SGD model loaded from disk.")
            return _sgd_model
        except Exception as exc:
            logger.warning("Failed to load SGD model, creating new: %s", exc)

    from sklearn.linear_model import SGDClassifier
    _sgd_model = SGDClassifier(
        loss="modified_huber",  # suporta predict_proba
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        n_jobs=1,
    )
    return _sgd_model


def partial_fit_example(features: list[float], label: int) -> None:
    """
    Atualiza o SGDClassifier com um único exemplo rotulado.
    Chamado imediatamente após cada /feedback do usuário.

    Args:
        features: vetor [avg_sentence_len, vocab_richness, burstiness,
                         punct_density, avg_word_len]
        label: 0 = humano, 1 = IA
    """
    import joblib
    import numpy as np

    model = _load_sgd()
    X = np.array([features])
    y = np.array([label])

    # classes= obrigatório na primeira chamada (SGD não conhece o espaço ainda)
    model.partial_fit(X, y, classes=[0, 1])

    joblib.dump(model, _SGD_PATH)
    logger.info("SGD partial_fit done: label=%d, features=%s", label, features)


def get_sgd_probability(features: list[float]) -> float | None:
    """
    Retorna probabilidade de ser IA pelo SGD.
    None se o modelo ainda não foi treinado com nenhum exemplo.
    """
    model = _load_sgd()
    if not hasattr(model, "classes_"):
        return None  # nunca recebeu partial_fit

    try:
        import numpy as np
        prob = model.predict_proba(np.array([features]))[0][1]
        return float(prob)
    except Exception as exc:
        logger.warning("SGD inference failed: %s", exc)
        return None


async def retrain_rf_with_examples(examples: list[dict]) -> bool:
    """
    Retreina o Random Forest com todos os exemplos confirmados.
    Destinado a ser chamado como background task do FastAPI.

    Args:
        examples: lista de dicts com 'features_json' (str) e 'label' (int)

    Returns:
        True se o retrain foi bem-sucedido.
    """
    if len(examples) < 20:
        logger.info("RF retrain skipped: only %d examples (need >= 20)", len(examples))
        return False

    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        X = np.array([json.loads(e["features_json"]) for e in examples])
        y = np.array([e["label"] for e in examples])

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        joblib.dump(model, _RF_PATH)
        logger.info(
            "RF retrain completed: %d examples, saved to %s",
            len(examples),
            _RF_PATH.name,
        )

        # Invalida cache do detection_service para carregar novo modelo
        import app.services.detection_service as ds
        ds._rf_model = None

        return True

    except Exception as exc:
        logger.error("RF retrain failed: %s", exc, exc_info=True)
        return False
