"""
Testes do serviço de aprendizado contínuo (learning_service).

Cobertura:
- partial_fit_example: treino online imediato
- get_sgd_probability: inferência antes e depois do treino
- retrain_rf_with_examples: retrain completo com exemplos suficientes e insuficientes
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services import learning_service

# Features válidas (12 dimensões — mesma ordem que extract_features)
HUMAN_FEATURES = [8.0, 0.72, 0.85, 0.04, 4.2, 0.005, 0.035, 0.020, 0.10, 0.08, 0.88, 0.62]
AI_FEATURES    = [22.0, 0.41, 0.22, 0.05, 5.8, 0.045, 0.003, 0.003, 0.00, 0.18, 0.76, 0.38]


class TestPartialFit:
    def test_partial_fit_human_label_does_not_crash(self, tmp_path):
        """partial_fit com label=0 (humano) não deve lançar exceção."""
        with patch.object(learning_service, "_SGD_PATH", tmp_path / "sgd_test.joblib"), \
             patch.object(learning_service, "_sgd_model", None):
            learning_service.partial_fit_example(HUMAN_FEATURES, 0)

    def test_partial_fit_ai_label_does_not_crash(self, tmp_path):
        """partial_fit com label=1 (IA) não deve lançar exceção."""
        with patch.object(learning_service, "_SGD_PATH", tmp_path / "sgd_test.joblib"), \
             patch.object(learning_service, "_sgd_model", None):
            learning_service.partial_fit_example(AI_FEATURES, 1)

    def test_partial_fit_saves_model_to_disk(self, tmp_path):
        """O modelo SGD deve ser salvo em disco após partial_fit."""
        sgd_path = tmp_path / "sgd_test.joblib"
        with patch.object(learning_service, "_SGD_PATH", sgd_path), \
             patch.object(learning_service, "_sgd_model", None):
            learning_service.partial_fit_example(AI_FEATURES, 1)
        assert sgd_path.exists(), "Modelo SGD não foi salvo"

    def test_partial_fit_multiple_calls_accumulate(self, tmp_path):
        """Múltiplos partial_fit devem acumular sem erro."""
        sgd_path = tmp_path / "sgd_test.joblib"
        with patch.object(learning_service, "_SGD_PATH", sgd_path), \
             patch.object(learning_service, "_sgd_model", None):
            for label in [0, 1, 0, 1, 0]:
                learning_service.partial_fit_example(AI_FEATURES, label)
        assert sgd_path.exists()

    def test_partial_fit_both_classes_required_before_predict(self, tmp_path):
        """SGD precisa ver ambas as classes antes de fazer predict_proba."""
        sgd_path = tmp_path / "sgd_test.joblib"
        with patch.object(learning_service, "_SGD_PATH", sgd_path), \
             patch.object(learning_service, "_sgd_model", None):
            # Após ver só classe 0
            learning_service.partial_fit_example(HUMAN_FEATURES, 0)
            # Ainda não deve ter classes_ completo para proba
            # (SGD vê classes=[0,1] na primeira chamada — deve ter ambas)
            learning_service.partial_fit_example(AI_FEATURES, 1)
            # Agora get_sgd_probability deve funcionar
            prob = learning_service.get_sgd_probability(AI_FEATURES)
            assert prob is not None
            assert 0.0 <= prob <= 1.0


class TestGetSGDProbability:
    def test_returns_none_when_model_untrained(self):
        """Antes de qualquer partial_fit, deve retornar None."""
        from sklearn.linear_model import SGDClassifier
        untrained = SGDClassifier(loss="modified_huber")
        with patch.object(learning_service, "_sgd_model", untrained):
            result = learning_service.get_sgd_probability(AI_FEATURES)
        assert result is None

    def test_returns_float_after_training(self, tmp_path):
        """Após treino com ambas as classes, deve retornar float."""
        sgd_path = tmp_path / "sgd_test.joblib"
        with patch.object(learning_service, "_SGD_PATH", sgd_path), \
             patch.object(learning_service, "_sgd_model", None):
            learning_service.partial_fit_example(HUMAN_FEATURES, 0)
            learning_service.partial_fit_example(AI_FEATURES, 1)
            prob = learning_service.get_sgd_probability(AI_FEATURES)
        assert isinstance(prob, float)

    def test_probability_between_0_and_1(self, tmp_path):
        sgd_path = tmp_path / "sgd_test.joblib"
        with patch.object(learning_service, "_SGD_PATH", sgd_path), \
             patch.object(learning_service, "_sgd_model", None):
            learning_service.partial_fit_example(HUMAN_FEATURES, 0)
            learning_service.partial_fit_example(AI_FEATURES, 1)
            prob = learning_service.get_sgd_probability(AI_FEATURES)
        assert 0.0 <= prob <= 1.0


class TestRetrainRF:
    @pytest.mark.asyncio
    async def test_retrain_skips_when_too_few_examples(self):
        """Menos de 20 exemplos deve pular o retrain."""
        examples = [
            {"features_json": json.dumps(AI_FEATURES), "label": 1}
            for _ in range(5)
        ]
        result = await learning_service.retrain_rf_with_examples(examples)
        assert result is False

    @pytest.mark.asyncio
    async def test_retrain_with_sufficient_examples(self, tmp_path):
        """Com >= 20 exemplos balanceados, retrain deve completar."""
        rf_path = tmp_path / "test_rf.joblib"
        examples = []
        for _ in range(15):
            examples.append({"features_json": json.dumps(HUMAN_FEATURES), "label": 0})
        for _ in range(15):
            examples.append({"features_json": json.dumps(AI_FEATURES), "label": 1})

        with patch.object(learning_service, "_RF_PATH", rf_path), \
             patch("app.services.learning_service.ds", MagicMock(), create=True):
            result = await learning_service.retrain_rf_with_examples(examples)

        assert result is True
        assert rf_path.exists(), "Modelo RF não foi salvo"

    @pytest.mark.asyncio
    async def test_retrain_resets_rf_cache(self, tmp_path):
        """Após retrain, o cache do detection_service deve ser invalidado."""
        import app.services.detection_service as ds_module
        ds_module._rf_model = MagicMock()  # simula modelo em cache

        rf_path = tmp_path / "test_rf.joblib"
        examples = []
        for _ in range(10):
            examples.append({"features_json": json.dumps(HUMAN_FEATURES), "label": 0})
            examples.append({"features_json": json.dumps(AI_FEATURES), "label": 1})

        with patch.object(learning_service, "_RF_PATH", rf_path):
            await learning_service.retrain_rf_with_examples(examples)

        # Cache deve ter sido resetado para None
        assert ds_module._rf_model is None

    @pytest.mark.asyncio
    async def test_retrain_handles_corrupted_features_gracefully(self):
        """features_json corrompido não deve crashar o retrain."""
        examples = [
            {"features_json": "INVALID_JSON", "label": 1}
            for _ in range(25)
        ]
        # Deve falhar graciosamente (retorna False) sem exceção não tratada
        try:
            result = await learning_service.retrain_rf_with_examples(examples)
            assert result is False
        except Exception:
            pytest.fail("retrain_rf_with_examples não deveria lançar exceção com JSON inválido")

    @pytest.mark.asyncio
    async def test_retrain_with_empty_list_returns_false(self):
        result = await learning_service.retrain_rf_with_examples([])
        assert result is False
