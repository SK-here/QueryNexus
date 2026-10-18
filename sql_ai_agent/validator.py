"""SQL safety validation before execution."""

from __future__ import annotations

import re

from sql_ai_agent.exceptions import SQLValidationError

# Statements that mutate or control the database
_WRITE_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|UPSERT|MERGE|REPLACE|"
    r"DROP|CREATE|ALTER|TRUNCATE|RENAME|"
    r"GRANT|REVOKE|"
    r"COPY|LOAD|CALL|EXECUTE|EXEC|"
    r"ATTACH|DETACH|"
    r"VACUUM|ANALYZE|REINDEX|"
    r"SET\s+ROLE|SET\s+SESSION"
    r")\b",
    re.IGNORECASE,
)

# Dangerous patterns even in "read" mode
_BLOCKED_ALWAYS = re.compile(
    r"\b("
    r"pg_sleep|sleep\s*\(|benchmark\s*\(|"
    r"load_file\s*\(|into\s+outfile|into\s+dumpfile|"
    r"xp_cmdshell|dblink_connect"
    r")\b",
    re.IGNORECASE,
)

_COMMENT_STRIP = re.compile(
    r"/\*.*?\*/|--.*?$|#.*?$",
    re.MULTILINE | re.DOTALL,
)


def _strip_comments(sql: str) -> str:
    return _COMMENT_STRIP.sub(" ", sql)


def extract_sql(raw: str) -> str:
    """Pull SQL out of markdown fences or prose wrappers from the LLM."""
    text = raw.strip()
    if not text:
        raise SQLValidationError("Model returned an empty response.")

    fence = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # If the model still wrapped with explanation, take the first SELECT/WITH/... line
    # Prefer keeping full text when it already looks like SQL.
    upper = text.lstrip().upper()
    if not any(
        upper.startswith(kw)
        for kw in (
            "SELECT",
            "WITH",
            "SHOW",
            "DESCRIBE",
            "DESC",
            "EXPLAIN",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "TRUNCATE",
        )
    ):
        # Try to find a SQL statement inside prose
        match = re.search(
            r"\b(WITH|SELECT|SHOW|DESCRIBE|DESC|EXPLAIN|INSERT|UPDATE|DELETE)\b[\s\S]+",
            text,
            re.IGNORECASE,
        )
        if match:
            text = match.group(0).strip()
        else:
            raise SQLValidationError(
                "Could not extract a SQL statement from the model response."
            )

    # Drop trailing semicolon for single-statement handling
    text = text.strip().rstrip(";").strip()
    if not text:
        raise SQLValidationError("Extracted SQL statement is empty.")
    return text


def validate_sql(sql: str, *, allow_writes: bool = False) -> str:
    """
    Validate SQL for safety constraints.

    Returns the cleaned SQL on success; raises SQLValidationError otherwise.
    """
    cleaned = extract_sql(sql)
    no_comments = _strip_comments(cleaned)

    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if len(statements) > 1:
        raise SQLValidationError(
            "Multiple SQL statements are not allowed. Send one statement at a time."
        )

    if _BLOCKED_ALWAYS.search(no_comments):
        raise SQLValidationError(
            "Query contains blocked functions or file I/O patterns."
        )

    if not allow_writes and _WRITE_KEYWORDS.search(no_comments):
        raise SQLValidationError(
            "Write/DDL statements are disabled. "
            "Re-run with allow_writes=True if you intentionally need mutations."
        )

    # Read-only mode: require a result-producing statement family
    if not allow_writes:
        head = no_comments.lstrip().upper()
        if not head.startswith(
            ("SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")
        ):
            raise SQLValidationError(
                "Only SELECT/WITH/SHOW/DESCRIBE/EXPLAIN statements are allowed "
                "when writes are disabled."
            )

    return cleaned
