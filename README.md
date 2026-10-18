# SQL AI Agent

Natural-language → SQL agent. Pass a database URL, user, and password; ask questions in plain English. The agent introspects the schema, generates SQL, validates it, executes it, and self-corrects on failures.

## Features

- Credentials via CLI flags or environment (`SQL_URL`, `SQL_USER`, `SQL_PASSWORD`)
- Schema-aware SQL generation (OpenAI-compatible LLM)
- Read-only by default (blocks `INSERT`/`UPDATE`/`DELETE`/DDL)
- Multi-statement and dangerous function blocking
- Automatic retries with the database error fed back to the model
- Ambiguous-question clarification instead of guessing
- Friendly connection / auth / timeout errors
- Result truncation with `max_rows`
- Postgres, MySQL/MariaDB, and SQLite

## Setup

```bash
cd QueryNexus/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set OPENAI_API_KEY (and optional SQL_* defaults) in .env
```

## Quick start (demo SQLite)

```bash
python scripts/seed_demo_db.py

python main.py ask "total paid revenue by country" \
  --sql-url "sqlite:///./demo.db" \
  --sql-user demo \
  --sql-password ""
```

Interactive mode:

```bash
python main.py chat \
  --sql-url "postgresql://localhost:5432/mydb" \
  --sql-user postgres \
  --sql-password 'secret'
```

One-shot against Postgres/MySQL:

```bash
export OPENAI_API_KEY=sk-...
python main.py ask "how many customers signed up last month?" \
  --sql-url "postgresql://localhost:5432/shop" \
  --sql-user app \
  --sql-password '***'
```

Generate SQL without running it:

```bash
python main.py ask "list unpaid orders" --no-execute \
  --sql-url "mysql://localhost:3306/shop" \
  --sql-user root \
  --sql-password '***'
```

## Python API

```python
from sql_ai_agent import SQLAIAgent

with SQLAIAgent(
    sql_url="postgresql://localhost:5432/shop",
    sql_user="app",
    sql_password="secret",
    allow_writes=False,
    max_rows=50,
) as agent:
    result = agent.ask("top 5 customers by revenue")
    if result.success:
        print(result.sql)
        print(result.rows)
    else:
        print(result.error_type, result.error)
        if result.clarification:
            print("Ask user:", result.clarification)
```

## Edge cases handled

| Situation | Behavior |
|-----------|----------|
| Missing URL / user | Clear validation error before connecting |
| Bad password / unreachable host | Mapped connection messages (auth, refused, timeout, unknown DB) |
| Empty / no tables | Schema error with guidance |
| Empty question | Rejected without calling the LLM |
| Ambiguous question | Clarification prompt (no unsafe guess) |
| Invalid SQL from model | Up to `MAX_RETRIES` self-repair loops |
| Write/DDL in read-only mode | `SQLValidationError` |
| Multiple statements | Blocked |
| Empty result set | Success with explanatory message |
| Large results | Truncated at `max_rows` |
| LLM API / rate limit errors | Surfaced as `SQLGenerationError` |

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required for generation |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `SQL_URL` / `SQL_USER` / `SQL_PASSWORD` | — | Optional defaults |
| `ALLOW_WRITES` | `false` | Permit mutations |
| `MAX_ROWS` | `100` | Result cap |
| `MAX_RETRIES` | `3` | Self-repair attempts |
| `QUERY_TIMEOUT_SECONDS` | `30` | Connect / statement timeout |

## Tests

```bash
pip install pytest
pytest -q
```

## Safety notes

- Defaults to **read-only** SQL. Pass `--allow-writes` only when you intend mutations.
- Prefer env vars or a secrets manager over putting passwords in shell history.
- The agent can still generate incorrect SQL; review queries in production workflows, especially with writes enabled.
