"""Interactive and one-shot CLI for the SQL AI agent."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sql_ai_agent.agent import AgentResponse, SQLAIAgent
from sql_ai_agent.config import get_settings

app = typer.Typer(
    add_completion=False,
    help="SQL AI Agent — ask questions in plain English, get SQL + results.",
)
console = Console()


def _resolve_credentials(
    sql_url: Optional[str],
    sql_user: Optional[str],
    sql_password: Optional[str],
) -> tuple[str, str, str]:
    settings = get_settings()
    url = (sql_url or settings.sql_url or "").strip()
    user = (sql_user or settings.sql_user or "").strip()
    # Explicit CLI empty string is allowed (local trust / SQLite)
    if sql_password is not None:
        password = sql_password
    else:
        password = settings.sql_password or ""

    missing = []
    if not url:
        missing.append("--sql-url / SQL_URL")
    if not user:
        missing.append("--sql-user / SQL_USER")
    if missing:
        console.print(
            f"[red]Missing required credentials:[/red] {', '.join(missing)}"
        )
        raise typer.Exit(code=2)
    return url, user, password


def _print_response(response: AgentResponse, *, as_json: bool = False) -> None:
    if as_json:
        console.print_json(data=response.to_dict())
        return

    if response.sql:
        console.print(Panel(response.sql, title="SQL", border_style="cyan"))

    if response.success:
        if response.columns:
            table = Table(show_header=True, header_style="bold")
            for col in response.columns:
                table.add_column(str(col))
            for row in response.rows:
                table.add_row(
                    *[
                        "" if row.get(c) is None else str(row.get(c))
                        for c in response.columns
                    ]
                )
            console.print(table)
            console.print(
                f"[dim]{response.row_count} row(s)"
                f"{' (truncated)' if response.truncated else ''}[/dim]"
            )
        elif response.message:
            console.print(f"[green]{response.message}[/green]")
        else:
            console.print(
                f"[green]OK[/green] — affected/returned row count: {response.row_count}"
            )
        if response.message and response.columns:
            console.print(f"[yellow]{response.message}[/yellow]")
        if response.retries_used:
            console.print(
                f"[dim]Self-corrected after {response.retries_used} retry(ies).[/dim]"
            )
    else:
        style = "yellow" if response.error_type == "AmbiguousQueryError" else "red"
        console.print(f"[{style}]{response.error_type}:[/{style}] {response.error}")
        if response.clarification:
            console.print(
                f"[yellow]Clarification needed:[/yellow] {response.clarification}"
            )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question about the data."),
    sql_url: Optional[str] = typer.Option(
        None, "--sql-url", envvar="SQL_URL", help="Database URL"
    ),
    sql_user: Optional[str] = typer.Option(
        None, "--sql-user", envvar="SQL_USER", help="Database user"
    ),
    sql_password: Optional[str] = typer.Option(
        None, "--sql-password", envvar="SQL_PASSWORD", help="Database password"
    ),
    allow_writes: bool = typer.Option(
        False, "--allow-writes", help="Permit INSERT/UPDATE/DELETE/DDL"
    ),
    no_execute: bool = typer.Option(
        False, "--no-execute", help="Generate SQL only; do not run it"
    ),
    max_rows: int = typer.Option(100, "--max-rows", help="Max rows to return"),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Ask one question and print SQL + results."""
    url, user, password = _resolve_credentials(sql_url, sql_user, sql_password)
    try:
        with SQLAIAgent(
            url,
            user,
            password,
            allow_writes=allow_writes,
            max_rows=max_rows,
        ) as agent:
            response = agent.ask(question, execute=not no_execute)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Fatal:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_response(response, as_json=as_json)
    raise typer.Exit(code=0 if response.success else 1)


@app.command("schema")
def show_schema(
    sql_url: Optional[str] = typer.Option(None, "--sql-url", envvar="SQL_URL"),
    sql_user: Optional[str] = typer.Option(None, "--sql-user", envvar="SQL_USER"),
    sql_password: Optional[str] = typer.Option(
        None, "--sql-password", envvar="SQL_PASSWORD"
    ),
) -> None:
    """Print the introspected database schema used for prompting."""
    url, user, password = _resolve_credentials(sql_url, sql_user, sql_password)
    try:
        with SQLAIAgent(url, user, password) as agent:
            console.print(agent.schema_summary())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Fatal:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def chat(
    sql_url: Optional[str] = typer.Option(None, "--sql-url", envvar="SQL_URL"),
    sql_user: Optional[str] = typer.Option(None, "--sql-user", envvar="SQL_USER"),
    sql_password: Optional[str] = typer.Option(
        None, "--sql-password", envvar="SQL_PASSWORD"
    ),
    allow_writes: bool = typer.Option(False, "--allow-writes"),
    max_rows: int = typer.Option(100, "--max-rows"),
) -> None:
    """Interactive REPL. Type questions; use /schema, /quit to control the session."""
    url, user, password = _resolve_credentials(sql_url, sql_user, sql_password)
    console.print(
        Panel(
            "SQL AI Agent ready. Type a question, or /schema /quit",
            title="chat",
            border_style="green",
        )
    )
    try:
        with SQLAIAgent(
            url, user, password, allow_writes=allow_writes, max_rows=max_rows
        ) as agent:
            while True:
                try:
                    line = console.input("[bold cyan]>[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\nBye.")
                    break
                if not line:
                    continue
                if line.lower() in {"/quit", "/exit", "quit", "exit"}:
                    console.print("Bye.")
                    break
                if line.lower() in {"/schema", "schema"}:
                    console.print(agent.schema_summary())
                    continue
                response = agent.ask(line)
                _print_response(response)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Fatal:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
