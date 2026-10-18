"""Domain-specific exceptions for the SQL AI agent."""


class SQLAgentError(Exception):
    """Base exception for all SQL agent failures."""

    def __init__(self, message: str, *, cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.cause = cause


class ConnectionError(SQLAgentError):
    """Database connection could not be established or was lost."""


class SchemaError(SQLAgentError):
    """Schema introspection failed or schema is unusable."""


class SQLGenerationError(SQLAgentError):
    """LLM failed to produce a usable SQL query."""


class SQLValidationError(SQLAgentError):
    """Generated SQL failed safety or syntax validation."""


class SQLExecutionError(SQLAgentError):
    """SQL executed but the database rejected it or returned an error."""


class AmbiguousQueryError(SQLAgentError):
    """User question is too ambiguous to map to a single SQL query."""


class EmptyResultError(SQLAgentError):
    """Query ran successfully but returned no rows."""
