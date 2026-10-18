"""SQL AI Agent — natural language to SQL with safe execution."""

from sql_ai_agent.agent import SQLAIAgent
from sql_ai_agent.exceptions import (
    ConnectionError,
    SchemaError,
    SQLGenerationError,
    SQLValidationError,
    SQLExecutionError,
    AmbiguousQueryError,
)

__all__ = [
    "SQLAIAgent",
    "ConnectionError",
    "SchemaError",
    "SQLGenerationError",
    "SQLValidationError",
    "SQLExecutionError",
    "AmbiguousQueryError",
]

__version__ = "1.0.0"
