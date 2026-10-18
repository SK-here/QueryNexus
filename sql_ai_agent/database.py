"""Database connection, schema introspection, and query execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus, urlparse, urlunparse

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import (
    DBAPIError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
    TimeoutError as SATimeoutError,
)

from sql_ai_agent.exceptions import ConnectionError, SchemaError, SQLExecutionError


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    primary_key: bool = False


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    foreign_keys: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SchemaInfo:
    dialect: str
    tables: list[TableInfo] = field(default_factory=list)

    def to_prompt(self) -> str:
        if not self.tables:
            return "No tables found in the connected database."

        parts = [f"Database dialect: {self.dialect}", ""]
        for table in self.tables:
            parts.append(f"Table: {table.name}")
            for col in table.columns:
                flags = []
                if col.primary_key:
                    flags.append("PK")
                if not col.nullable:
                    flags.append("NOT NULL")
                flag_str = f" ({', '.join(flags)})" if flags else ""
                parts.append(f"  - {col.name}: {col.type}{flag_str}")
            for fk in table.foreign_keys:
                constrained = ", ".join(fk.get("constrained_columns") or [])
                referred_table = fk.get("referred_table", "?")
                referred_cols = ", ".join(fk.get("referred_columns") or [])
                parts.append(
                    f"  FK: ({constrained}) -> {referred_table}({referred_cols})"
                )
            parts.append("")
        return "\n".join(parts).strip()


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False


class DatabaseManager:
    """Manages SQLAlchemy engine lifecycle and schema/query operations."""

    def __init__(
        self,
        sql_url: str,
        sql_user: str,
        sql_password: str,
        *,
        query_timeout_seconds: int = 30,
        max_rows: int = 100,
    ):
        if not sql_url or not sql_url.strip():
            raise ConnectionError("SQL URL is required and cannot be empty.")
        if not sql_user or not sql_user.strip():
            raise ConnectionError("SQL user is required and cannot be empty.")
        # Password may be empty for local trust auth, but must be a string
        if sql_password is None:
            raise ConnectionError("SQL password must be provided (use '' if none).")

        self.sql_url = sql_url.strip()
        self.sql_user = sql_user.strip()
        self.sql_password = sql_password
        self.query_timeout_seconds = query_timeout_seconds
        self.max_rows = max_rows
        self._engine: Engine | None = None
        self._schema: SchemaInfo | None = None

    @staticmethod
    def build_connection_url(sql_url: str, sql_user: str, sql_password: str) -> str:
        """
        Build a SQLAlchemy URL from a host URL plus credentials.

        Accepts:
          - Full URLs: postgresql://user:pass@host:5432/db
          - Host-style: postgresql://localhost:5432/db  (user/pass injected)
          - Driver shortcuts: postgresql+psycopg2://..., mysql+pymysql://...
        """
        url = sql_url.strip()

        # Expand common short schemes
        scheme_map = {
            "postgres://": "postgresql+psycopg2://",
            "postgresql://": "postgresql+psycopg2://",
            "mysql://": "mysql+pymysql://",
            "mariadb://": "mysql+pymysql://",
        }
        for short, full in scheme_map.items():
            if url.lower().startswith(short):
                url = full + url[len(short) :]
                break

        parsed = urlparse(url)
        if not parsed.scheme:
            raise ConnectionError(
                "SQL URL must include a scheme "
                "(e.g. postgresql://host:5432/dbname or mysql://host:3306/dbname)."
            )
        if not parsed.hostname and not parsed.path:
            raise ConnectionError(
                f"Could not parse host/database from SQL URL: {sql_url!r}"
            )

        # SQLite is file-based — do not inject network credentials into the URL
        if parsed.scheme.startswith("sqlite"):
            return url

        # If credentials already in URL, keep them unless caller overrides
        username = parsed.username or sql_user
        password = parsed.password if parsed.password is not None else sql_password

        # Rebuild netloc with encoded credentials
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        userinfo = f"{quote_plus(username)}:{quote_plus(password)}" if username else ""
        netloc = f"{userinfo}@{host}{port}" if userinfo else f"{host}{port}"

        return urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    def connect(self) -> Engine:
        if self._engine is not None:
            return self._engine

        connection_url = self.build_connection_url(
            self.sql_url, self.sql_user, self.sql_password
        )
        try:
            engine = create_engine(
                connection_url,
                pool_pre_ping=True,
                pool_recycle=1800,
                connect_args=self._connect_args(connection_url),
            )
            # Verify connectivity eagerly
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._engine = engine
            return engine
        except OperationalError as exc:
            raise ConnectionError(
                self._friendly_connection_message(exc), cause=exc
            ) from exc
        except SQLAlchemyError as exc:
            raise ConnectionError(
                f"Failed to connect to the database: {exc}", cause=exc
            ) from exc
        except Exception as exc:  # noqa: BLE001 — surface unexpected connect failures
            raise ConnectionError(
                f"Unexpected connection failure: {exc}", cause=exc
            ) from exc

    def _connect_args(self, connection_url: str) -> dict[str, Any]:
        args: dict[str, Any] = {}
        lower = connection_url.lower()
        if lower.startswith("postgresql"):
            args["connect_timeout"] = self.query_timeout_seconds
            args["options"] = f"-c statement_timeout={self.query_timeout_seconds * 1000}"
        elif lower.startswith("mysql"):
            args["connect_timeout"] = self.query_timeout_seconds
            args["read_timeout"] = self.query_timeout_seconds
            args["write_timeout"] = self.query_timeout_seconds
        elif lower.startswith("sqlite"):
            # SQLite has no network timeout; ignore
            pass
        return args

    @staticmethod
    def _friendly_connection_message(exc: OperationalError) -> str:
        msg = str(exc.orig if hasattr(exc, "orig") and exc.orig else exc).lower()
        if "password authentication failed" in msg or "access denied" in msg:
            return "Authentication failed. Check SQL user and password."
        if "could not connect" in msg or "connection refused" in msg:
            return (
                "Could not reach the database host. "
                "Check the SQL URL host/port and that the server is running."
            )
        if "timeout" in msg or "timed out" in msg:
            return "Connection timed out. The database may be unreachable or overloaded."
        if "does not exist" in msg or "unknown database" in msg:
            return "Database name in the SQL URL was not found on the server."
        if "ssl" in msg:
            return f"SSL/TLS connection error: {exc}"
        return f"Database connection failed: {exc}"

    @property
    def engine(self) -> Engine:
        return self.connect()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._schema = None

    def __enter__(self) -> DatabaseManager:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_schema(self, *, refresh: bool = False) -> SchemaInfo:
        if self._schema is not None and not refresh:
            return self._schema

        try:
            inspector = inspect(self.engine)
            dialect = self.engine.dialect.name
            table_names = inspector.get_table_names()
            # Include views as read-only sources
            try:
                view_names = inspector.get_view_names()
            except NotImplementedError:
                view_names = []

            tables: list[TableInfo] = []
            for name in sorted(set(table_names) | set(view_names)):
                columns_raw = inspector.get_columns(name)
                pk = set(inspector.get_pk_constraint(name).get("constrained_columns") or [])
                try:
                    fks = inspector.get_foreign_keys(name)
                except NotImplementedError:
                    fks = []

                columns = [
                    ColumnInfo(
                        name=c["name"],
                        type=str(c.get("type", "UNKNOWN")),
                        nullable=bool(c.get("nullable", True)),
                        primary_key=c["name"] in pk,
                    )
                    for c in columns_raw
                ]
                tables.append(TableInfo(name=name, columns=columns, foreign_keys=fks))

            if not tables:
                raise SchemaError(
                    "Connected successfully, but no tables or views were found. "
                    "Check that you are connected to the correct database/schema."
                )

            self._schema = SchemaInfo(dialect=dialect, tables=tables)
            return self._schema
        except SchemaError:
            raise
        except SQLAlchemyError as exc:
            raise SchemaError(f"Failed to introspect schema: {exc}", cause=exc) from exc

    def execute(self, sql: str, *, max_rows: int | None = None) -> QueryResult:
        limit = max_rows if max_rows is not None else self.max_rows
        cleaned = sql.strip().rstrip(";")
        if not cleaned:
            raise SQLExecutionError("Cannot execute an empty SQL statement.")

        try:
            with self.engine.connect() as conn:
                # Use a transaction that we always roll back for safety on writes
                # unless the statement is clearly a SELECT (auto-commit not needed).
                result: Result = conn.execute(text(cleaned))

                if not result.returns_rows:
                    conn.commit()
                    return QueryResult(
                        columns=[],
                        rows=[],
                        row_count=result.rowcount if result.rowcount is not None else 0,
                        truncated=False,
                    )

                columns = list(result.keys())
                fetched = result.fetchmany(limit + 1)
                truncated = len(fetched) > limit
                rows_raw = fetched[:limit]
                rows = [dict(zip(columns, row)) for row in rows_raw]
                # Ensure SELECT side-effects aren't held
                conn.rollback()
                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    truncated=truncated,
                )
        except SATimeoutError as exc:
            raise SQLExecutionError(
                f"Query timed out after {self.query_timeout_seconds}s.", cause=exc
            ) from exc
        except ProgrammingError as exc:
            raise SQLExecutionError(
                f"SQL syntax or semantic error: {self._db_error_detail(exc)}",
                cause=exc,
            ) from exc
        except OperationalError as exc:
            raise SQLExecutionError(
                f"Database operational error: {self._db_error_detail(exc)}",
                cause=exc,
            ) from exc
        except DBAPIError as exc:
            raise SQLExecutionError(
                f"Database error: {self._db_error_detail(exc)}", cause=exc
            ) from exc
        except SQLAlchemyError as exc:
            raise SQLExecutionError(f"Failed to execute SQL: {exc}", cause=exc) from exc

    @staticmethod
    def _db_error_detail(exc: Exception) -> str:
        orig = getattr(exc, "orig", None)
        detail = str(orig) if orig else str(exc)
        # Collapse noisy multiline driver messages
        detail = re.sub(r"\s+", " ", detail).strip()
        return detail[:500]
