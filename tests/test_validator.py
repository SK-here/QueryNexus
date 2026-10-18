import pytest

from sql_ai_agent.exceptions import SQLValidationError
from sql_ai_agent.validator import extract_sql, validate_sql


def test_extract_from_fence():
    raw = "Sure!\n```sql\nSELECT 1;\n```\n"
    assert extract_sql(raw) == "SELECT 1"


def test_blocks_writes_by_default():
    with pytest.raises(SQLValidationError):
        validate_sql("DELETE FROM customers")


def test_allows_writes_when_enabled():
    sql = validate_sql("DELETE FROM customers WHERE id = 1", allow_writes=True)
    assert sql.upper().startswith("DELETE")


def test_blocks_multiple_statements():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT 1; DROP TABLE customers")


def test_blocks_dangerous_functions():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT pg_sleep(10)")
