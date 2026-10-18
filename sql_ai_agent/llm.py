"""Thin OpenAI-compatible chat client for SQL generation."""

from __future__ import annotations

import json
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from sql_ai_agent.exceptions import SQLGenerationError


SYSTEM_PROMPT = """\
You are an expert SQL generator. Convert the user's natural-language question \
into a single SQL query for the given database schema.

Rules:
1. Output ONLY the SQL query — no markdown fences, no commentary, no preamble.
2. Use only tables and columns that exist in the provided schema.
3. Prefer ANSI SQL compatible with the stated dialect; use dialect-specific \
features only when necessary.
4. Never invent tables, columns, or joins that are not supported by the schema.
5. If the question is ambiguous or cannot be answered from the schema, respond \
with exactly this JSON (and nothing else):
   {"error":"ambiguous","message":"<brief clarification question>"}
6. Do not use multiple statements. Do not use destructive operations unless \
the user explicitly asks and writes are allowed.
7. Limit large result sets with a reasonable LIMIT (default 100) when the user \
does not specify a count.
8. For aggregations, use clear aliases.
9. Qualify column names with table aliases when joining.
"""


class LLMClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        if not api_key or not api_key.strip():
            raise SQLGenerationError(
                "OPENAI_API_KEY is required. Set it in the environment or .env file."
            )
        self.model = model
        self._client = OpenAI(
            api_key=api_key.strip(),
            base_url=base_url,
            timeout=timeout,
        )

    def generate_sql(
        self,
        question: str,
        schema_prompt: str,
        *,
        allow_writes: bool = False,
        dialect: str = "unknown",
        previous_sql: str | None = None,
        previous_error: str | None = None,
    ) -> str:
        write_policy = (
            "Write/DDL statements ARE allowed for this request."
            if allow_writes
            else "Write/DDL statements are FORBIDDEN. Produce read-only SQL only."
        )

        user_parts = [
            f"Dialect: {dialect}",
            write_policy,
            "",
            "Schema:",
            schema_prompt,
            "",
            f"Question: {question}",
        ]

        if previous_sql and previous_error:
            user_parts.extend(
                [
                    "",
                    "The previous SQL failed. Fix it and return corrected SQL only.",
                    f"Previous SQL:\n{previous_sql}",
                    f"Database error:\n{previous_error}",
                ]
            )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(user_parts)},
                ],
            )
        except RateLimitError as exc:
            raise SQLGenerationError(
                "LLM rate limit exceeded. Wait and retry.", cause=exc
            ) from exc
        except APITimeoutError as exc:
            raise SQLGenerationError(
                "LLM request timed out.", cause=exc
            ) from exc
        except APIConnectionError as exc:
            raise SQLGenerationError(
                "Could not reach the LLM API. Check network / OPENAI_BASE_URL.",
                cause=exc,
            ) from exc
        except APIStatusError as exc:
            raise SQLGenerationError(
                f"LLM API error ({exc.status_code}): {exc.message}", cause=exc
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise SQLGenerationError(
                f"Unexpected LLM failure: {exc}", cause=exc
            ) from exc

        choice = response.choices[0].message.content if response.choices else None
        if not choice or not choice.strip():
            raise SQLGenerationError("LLM returned an empty completion.")

        return choice.strip()

    @staticmethod
    def parse_ambiguity(content: str) -> str | None:
        """Return clarification message if the model signaled ambiguity."""
        text = content.strip()
        if not text.startswith("{"):
            return None
        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            return None
        if data.get("error") == "ambiguous":
            return str(data.get("message") or "Please clarify your question.")
        return None
