"""
GET /api/v1/metrics — informações sobre o modelo ML ativo.
"""

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/metrics", tags=["metrics"])

_ML_DIR = Path(__file__).resolve().parents[4] / "app" / "models" / "ml"


class ModelMetrics(BaseModel):
    model_version: str
    deploy_model: str
    n_features: int
    feature_names: list[str]
    n_train_samples: int
    n_test_samples: int
    datasets_used: list[str]
    test_accuracy: float
    test_f1_weighted: float
    test_roc_auc: float
    test_brier: float
    optuna_best_auc: float | None = None
    model_size_mb: float
    model_file: str


@router.get("", response_model=ModelMetrics, summary="Métricas do modelo ML ativo")
def get_metrics():
    """
    Retorna as métricas de desempenho e configuração do modelo de detecção ativo.

    - **test_accuracy**: acurácia no conjunto de teste
    - **test_roc_auc**: AUC-ROC (quanto maior, melhor — 0.5 = aleatório, 1.0 = perfeito)
    - **test_brier**: Brier score (quanto menor, melhor — calibração de probabilidade)
    - **n_features**: número de features usadas pelo modelo
    """
    # Tenta v4 primeiro, cai para v3
    for metrics_file, model_file, version in [
        (_ML_DIR / "v4_metrics.json",    _ML_DIR / "detector_v4_best.joblib", "v4"),
        (_ML_DIR / "v3_metrics.json",    _ML_DIR / "detector_v3_rf.joblib",   "v3"),
    ]:
        if metrics_file.exists() and model_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                m = json.load(f)

            size_mb = round(model_file.stat().st_size / 1024 / 1024, 2)

            # v3 usa campos diferentes — normaliza
            if version == "v3":
                return ModelMetrics(
                    model_version="v3",
                    deploy_model="RandomForestClassifier",
                    n_features=m.get("n_features", 12),
                    feature_names=list(m.get("feature_importances", {}).keys()),
                    n_train_samples=m.get("n_train_samples", 0),
                    n_test_samples=0,
                    datasets_used=["hc3"],
                    test_accuracy=m.get("cv_accuracy_mean", 0.0),
                    test_f1_weighted=0.0,
                    test_roc_auc=0.0,
                    test_brier=0.0,
                    model_size_mb=size_mb,
                    model_file=model_file.name,
                )

            return ModelMetrics(
                model_version=m["model_version"],
                deploy_model=m["deploy_model"],
                n_features=m["n_features"],
                feature_names=m["feature_names"],
                n_train_samples=m["n_train_samples"],
                n_test_samples=m["n_test_samples"],
                datasets_used=m["datasets_used"],
                test_accuracy=m["test_accuracy"],
                test_f1_weighted=m["test_f1_weighted"],
                test_roc_auc=m["test_roc_auc"],
                test_brier=m["test_brier"],
                optuna_best_auc=m.get("optuna_best_auc"),
                model_size_mb=size_mb,
                model_file=model_file.name,
            )

    raise HTTPException(status_code=503, detail="Nenhum modelo ML disponível.")
