# AI Detector

Plataforma para detectar se um texto foi gerado ou editado por Inteligência Artificial.

## Stack

- **FastAPI** — API REST assíncrona com documentação automática
- **SQLAlchemy 2.0** — ORM com suporte nativo async
- **Alembic** — Migrations de banco de dados
- **SQLite** — Banco de dados para desenvolvimento
- **Pydantic v2** — Validação de dados e schemas

## Arquitetura

```
HTTP Request → [Router] → [Service] → [Repository] → [ORM Model]
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
| POST | `/api/v1/submissions` | Criar submissão de texto |
| GET | `/api/v1/submissions` | Listar submissões |
| GET | `/api/v1/submissions/{id}` | Detalhe da submissão |
| PUT | `/api/v1/submissions/{id}` | Atualizar submissão |
| DELETE | `/api/v1/submissions/{id}` | Deletar submissão |
| POST | `/api/v1/submissions/{id}/analyze` | Disparar análise de IA |
| GET | `/api/v1/submissions/{id}/result` | Obter resultado da análise |

## Rodando os testes

```bash
pytest --cov=app tests/
```
