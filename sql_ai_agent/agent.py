"""High-level SQL AI agent: NL → validated SQL → execute with retries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sql_ai_agent.config import Settings, get_settings
from sql_ai_agent.database import DatabaseManager
from sql_ai_agent.exceptions import (
    AmbiguousQueryError,
    ConnectionError,
    EmptyResultError,
    SQLExecutionError,
    SQLGenerationError,
    SQLValidationError,
)
from sql_ai_agent.llm import LLMClient
from sql_ai_agent.validator import validate_sql


@dataclass
class AgentResponse:
    """Structured result returned to callers (CLI / API)."""

    success: bool
    question: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    retries_used: int = 0
    error: str | None = None
    error_type: str | None = None
    clarification: str | None = None
    message: str | None = None
    general_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "question": self.question,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "retries_used": self.retries_used,
            "error": self.error,
            "error_type": self.error_type,
            "clarification": self.clarification,
            "message": self.message,
            "general_answer": self.general_answer,
        }


class SQLAIAgent:
    """
    Natural-language SQL agent.

    Parameters
    ----------
    sql_url:
        Database URL, e.g. ``postgresql://localhost:5432/mydb`` or
        ``mysql://localhost:3306/mydb``.
    sql_user / sql_password:
        Credentials injected into the connection URL when not already present.
    allow_writes:
        When False (default), only read-only statements are accepted.
    """

    def __init__(
        self,
        sql_url: str,
        sql_user: str,
        sql_password: str,
        *,
        settings: Settings | None = None,
        allow_writes: bool | None = None,
        max_retries: int | None = None,
        max_rows: int | None = None,
        llm: LLMClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.allow_writes = (
            allow_writes
            if allow_writes is not None
            else self.settings.allow_writes
        )
        self.max_retries = (
            max_retries if max_retries is not None else self.settings.max_retries
        )
        self.max_rows = max_rows if max_rows is not None else self.settings.max_rows

        self.db = DatabaseManager(
            sql_url=sql_url,
            sql_user=sql_user,
            sql_password=sql_password,
            query_timeout_seconds=self.settings.query_timeout_seconds,
            max_rows=self.max_rows,
        )
        provider = getattr(self.settings, "llm_provider", "openai")
        effective_key = self.settings.get_effective_api_key(provider)
        effective_model = self.settings.get_effective_model(provider) or None
        effective_base_url = self.settings.get_effective_base_url(provider) or None

        self.llm = llm or LLMClient(
            api_key=effective_key,
            provider=provider,
            model=effective_model,
            base_url=effective_base_url,
        )
        self._connected = False

    def connect(self) -> None:
        """Eagerly connect and cache schema."""
        self.db.connect()
        self.db.get_schema()
        self._connected = True

    def close(self) -> None:
        self.db.close()
        self._connected = False

    def __enter__(self) -> SQLAIAgent:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def ask(
        self,
        question: str,
        *,
        execute: bool = True,
        raise_on_empty: bool = False,
        allow_general: bool = False,
    ) -> AgentResponse:
        """
        Convert *question* to SQL, optionally execute it, and return results.

        Handles connection, generation, validation, execution, and self-repair
        retries. Never raises for expected domain failures — they are returned
        as ``AgentResponse(success=False, ...)``. Unexpected bugs still raise.
        """
        question = (question or "").strip()
        if not question:
            return AgentResponse(
                success=False,
                question=question,
                error="Question cannot be empty.",
                error_type="EmptyQuestion",
            )

        try:
            if not self._connected:
                self.connect()
            schema = self.db.get_schema()
        except ConnectionError as exc:
            return AgentResponse(
                success=False,
                question=question,
                error=exc.message,
                error_type="ConnectionError",
            )
        except Exception as exc:  # schema / connect edge cases
            return AgentResponse(
                success=False,
                question=question,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        sql: str | None = None
        last_error: str | None = None
        retries_used = 0

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm.generate_sql(
                    question,
                    schema.to_prompt(),
                    allow_writes=self.allow_writes,
                    dialect=schema.dialect,
                    previous_sql=sql,
                    previous_error=last_error,
                )

                clarification = self.llm.parse_ambiguity(raw)
                if clarification:
                    if allow_general:
                        answer = self.llm.generate_general_answer(question)
                        return AgentResponse(
                            success=True,
                            question=question,
                            general_answer=answer,
                            message=(
                                "Answered from the model's general knowledge — "
                                "this question could not be answered from the connected database."
                            ),
                            retries_used=retries_used,
                        )
                    return AgentResponse(
                        success=False,
                        question=question,
                        error="Question is ambiguous.",
                        error_type="AmbiguousQueryError",
                        clarification=clarification,
                        retries_used=retries_used,
                    )

                sql = validate_sql(raw, allow_writes=self.allow_writes)

                if not execute:
                    return AgentResponse(
                        success=True,
                        question=question,
                        sql=sql,
                        retries_used=retries_used,
                        message="SQL generated (not executed).",
                    )

                result = self.db.execute(sql, max_rows=self.max_rows)

                if raise_on_empty and result.row_count == 0 and result.columns:
                    raise EmptyResultError("Query returned no rows.")

                message = None
                if result.row_count == 0 and result.columns:
                    message = "Query succeeded but returned no rows."
                elif result.truncated:
                    message = (
                        f"Result truncated to {self.max_rows} rows. "
                        "Narrow the question or raise max_rows."
                    )

                return AgentResponse(
                    success=True,
                    question=question,
                    sql=sql,
                    columns=result.columns,
                    rows=result.rows,
                    row_count=result.row_count,
                    truncated=result.truncated,
                    retries_used=retries_used,
                    message=message,
                )

            except AmbiguousQueryError as exc:
                return AgentResponse(
                    success=False,
                    question=question,
                    sql=sql,
                    error=exc.message,
                    error_type="AmbiguousQueryError",
                    clarification=exc.message,
                    retries_used=retries_used,
                )
            except SQLValidationError as exc:
                # Validation failures are usually model mistakes — retry with feedback
                last_error = exc.message
                retries_used = attempt + 1
                if attempt >= self.max_retries:
                    return AgentResponse(
                        success=False,
                        question=question,
                        sql=sql,
                        error=exc.message,
                        error_type="SQLValidationError",
                        retries_used=retries_used,
                    )
            except SQLExecutionError as exc:
                last_error = exc.message
                retries_used = attempt + 1
                if attempt >= self.max_retries:
                    return AgentResponse(
                        success=False,
                        question=question,
                        sql=sql,
                        error=(
                            f"SQL failed after {retries_used} attempt(s): {exc.message}"
                        ),
                        error_type="SQLExecutionError",
                        retries_used=retries_used,
                    )
            except SQLGenerationError as exc:
                return AgentResponse(
                    success=False,
                    question=question,
                    sql=sql,
                    error=exc.message,
                    error_type="SQLGenerationError",
                    retries_used=retries_used,
                )
            except EmptyResultError as exc:
                return AgentResponse(
                    success=False,
                    question=question,
                    sql=sql,
                    columns=[],
                    rows=[],
                    row_count=0,
                    error=exc.message,
                    error_type="EmptyResultError",
                    retries_used=retries_used,
                )
            except ConnectionError as exc:
                return AgentResponse(
                    success=False,
                    question=question,
                    sql=sql,
                    error=exc.message,
                    error_type="ConnectionError",
                    retries_used=retries_used,
                )
            except Exception as exc:  # noqa: BLE001 — last-resort boundary
                return AgentResponse(
                    success=False,
                    question=question,
                    sql=sql,
                    error=f"Unexpected error: {exc}",
                    error_type=type(exc).__name__,
                    retries_used=retries_used,
                )

        return AgentResponse(
            success=False,
            question=question,
            sql=sql,
            error=last_error or "Failed to produce a working SQL query.",
            error_type="SQLGenerationError",
            retries_used=retries_used,
        )

    def schema_summary(self) -> str:
        if not self._connected:
            self.connect()
        return self.db.get_schema().to_prompt()
