# AI Detector

Plataforma para detectar se um texto foi gerado ou editado por Inteligência Artificial.

## Stack

- **FastAPI** — API REST assíncrona com documentação automática
- **SQLAlchemy 2.0** — ORM com suporte nativo async
- **Alembic** — Migrations de banco de dados
- **SQLite** — Banco de dados para desenvolvimento
- **Pydantic v2** — Validação de dados e schemas
- **XGBoost + scikit-learn** — Modelo ML de detecção
- **Optuna** — Otimização de hiperparâmetros

## Arquitetura

```
HTTP Request -> [Router] -> [Service] -> [Repository] -> [ORM Model]
```

### Pipeline de detecção (3 camadas)

```
Texto -> Camada 1: Heurísticas (20 features)
      -> Camada 2: XGBoost v4 (score ML)
      -> Camada 3: Claude Haiku (zona incerta 0.35-0.65)
      -> Veredicto: human | ai | uncertain
```

## Rodando o projeto

```bash
# 1. Criar ambiente virtual
python -m venv .venv
source .venv/Scripts/activate  # Windows

# 2. Instalar dependências
pip install -r requirements-dev.txt

# 3. Configurar variáveis de ambiente
cp .env.example .env

# 4. Aplicar migrations
alembic upgrade head

# 5. Rodar o servidor (porta 8001)
uvicorn app.main:app --reload --port 8001
```

Acesse a documentação em: http://localhost:8001/docs

## Endpoints

| Método | Caminho | Descrição |
|---|---|---|
| POST | `/api/v1/detect` | Detectar IA em texto ou imagem |
| POST | `/api/v1/detect/feedback` | Corrigir resultado e treinar modelo |
| GET | `/api/v1/metrics` | Métricas e info do modelo ML ativo |
| POST | `/api/v1/submissions` | Criar submissão de texto |
| GET | `/api/v1/submissions` | Listar submissões |
| GET | `/api/v1/submissions/{id}` | Detalhe da submissão |
| PUT | `/api/v1/submissions/{id}` | Atualizar submissão |
| DELETE | `/api/v1/submissions/{id}` | Deletar submissão |
| POST | `/api/v1/submissions/{id}/analyze` | Disparar análise de IA |
| GET | `/api/v1/submissions/{id}/result` | Obter resultado da análise |

## Modelo ML — v4

O modelo de detecção atual é um **XGBoost calibrado com Platt Scaling**, treinado com 237.693 textos de 4 datasets públicos.

### Datasets de treino

| Dataset | Tamanho (amostras) | Descrição |
|---|---|---|
| [RAID](https://huggingface.co/datasets/liamdugan/raid) | 50K | Benchmark adversarial — múltiplos LLMs e ataques |
| [AI Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile) | 30K | Textos humanos curados |
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | 84K | Perguntas com respostas humanas vs ChatGPT |
| [MAGE](https://huggingface.co/datasets/yaful/MAGE) | 80K | Textos humanos vs múltiplos LLMs |

### Features (20)

**Grupo 1 — Estruturais (12):** avg_sentence_length, vocabulary_richness, burstiness, punctuation_density, avg_word_length, transition_word_density, first_person_ratio, hedge_word_ratio, question_density, bigram_repetition_score, lexical_diversity_entropy, hapax_legomena_ratio

**Grupo 2 — Estilométricas (8):** readability_fog, stopword_ratio, sentence_length_variance, comma_density, unique_trigrams_ratio, pos_noun_ratio, coherence_score, exclamation_ratio

### Métricas (conjunto de teste — 47.539 amostras)

| Métrica | Valor |
|---|---|
| Accuracy | 78.45% |
| F1 (weighted) | 0.784 |
| AUC-ROC | **0.865** |
| Brier Score | 0.164 |
| Optuna best AUC | 0.867 (100 trials) |

### Comparativo de versões

| Versão | Algoritmo | Treino | Accuracy | AUC |
|---|---|---|---|---|
| v1 | SGD online | HC3 (5K) | ~70% | — |
| v2 | RandomForest | HC3 (8K) | ~85%* | — |
| v3 | RandomForest | HC3 (16K) | 93.4%* | — |
| **v4** | **XGBoost + Platt** | **RAID+HC3+MAGE+Pile (237K)** | **78.5%** | **0.865** |

*v2/v3: métricas medidas apenas no HC3 (distribuição simples). v4 é avaliado em dados diversos e adversariais — mais representativo do mundo real.

## Treinando o modelo

```bash
# Treino completo (100 trials Optuna, ~1h)
python scripts/train_v4.py

# Treino rapido com hiperparametros pre-otimizados (~3 min)
python scripts/train_v4.py --fast

# Ativar modelo treinado no servico de deteccao
python scripts/update_service_v4.py
```

O script salva checkpoint de features em `app/models/ml/v4_features_checkpoint.npz`.
Se interrompido, o próximo run pula a extração de features automaticamente.

## Rodando os testes

```bash
pytest --cov=app tests/
```

249 testes — auth, detecção, segurança.
