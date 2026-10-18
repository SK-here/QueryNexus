from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from sql_ai_agent.database import DatabaseManager
from sql_ai_agent.exceptions import ConnectionError, SQLExecutionError


@pytest.fixture()
def sqlite_url(tmp_path: Path) -> str:
    db = tmp_path / "t.db"
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("INSERT INTO items (id, name) VALUES (1, 'alpha'), (2, 'beta')"))
    return f"sqlite:///{db}"


def test_schema_and_query(sqlite_url: str):
    with DatabaseManager(sqlite_url, "u", "") as db:
        schema = db.get_schema()
        assert schema.dialect == "sqlite"
        assert any(t.name == "items" for t in schema.tables)
        result = db.execute("SELECT name FROM items ORDER BY id")
        assert result.row_count == 2
        assert result.rows[0]["name"] == "alpha"


def test_empty_url_rejected():
    with pytest.raises(ConnectionError):
        DatabaseManager("", "u", "p")


def test_bad_sql(sqlite_url: str):
    with DatabaseManager(sqlite_url, "u", "") as db:
        with pytest.raises(SQLExecutionError):
            db.execute("SELECT * FROM does_not_exist")
