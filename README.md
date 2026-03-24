# Python Projects

Repositório de projetos Python focados em Machine Learning e detecção de IA.

## Projetos

### `machine-learning/ai-detector`

Plataforma para detectar se um texto foi gerado ou editado por Inteligência Artificial.

**Stack:** FastAPI · SQLAlchemy 2.0 · Alembic · SQLite · Pydantic v2

**Funcionalidades:**
- API REST assíncrona para análise de textos
- Arquitetura em camadas (Router → Service → Repository → ORM)
- Migrations com Alembic
- Testes unitários com cobertura

**Rodar:**
```bash
cd machine-learning/ai-detector
pip install -r requirements-dev.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8001
```
Documentação: `http://localhost:8001/docs`

---

## Licença

MIT
