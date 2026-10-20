"""Multi-provider LLM client for SQL generation (OpenAI, Gemini, Groq, DeepSeek, Ollama, Anthropic, Mistral, OpenRouter)."""

from __future__ import annotations

import json
from typing import Any, Dict

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

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

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini", "gpt-4-turbo"],
        "requires_key": True,
        "env_var": "OPENAI_API_KEY",
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "requires_key": True,
        "env_var": "GEMINI_API_KEY",
    },
    "groq": {
        "name": "Groq Cloud",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-20b",
        "models": [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "minimaxai/minimax-m2.7",
            "meta-models/muse-spark-open",
            "openai/gpt-oss-safeguard-20b",
            # Legacy enterprise-only models (removed for free/developer tiers 2026-08-16)
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ],
        "requires_key": True,
        "env_var": "GROQ_API_KEY",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "requires_key": True,
        "env_var": "DEEPSEEK_API_KEY",
    },
    "ollama": {
        "name": "Ollama (Local / Free)",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.2",
        "models": ["llama3.2", "qwen2.5-coder", "mistral", "deepseek-r1:8b", "codellama"],
        "requires_key": False,
        "env_var": "OLLAMA_API_KEY",
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-3-5-sonnet-20241022",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "requires_key": True,
        "env_var": "ANTHROPIC_API_KEY",
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "anthropic/claude-3.5-sonnet",
        "models": ["anthropic/claude-3.5-sonnet", "meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-r1"],
        "requires_key": True,
        "env_var": "OPENROUTER_API_KEY",
    },
    "mistral": {
        "name": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
        "models": ["mistral-small-latest", "codestral-latest", "mistral-large-latest"],
        "requires_key": True,
        "env_var": "MISTRAL_API_KEY",
    },
    "custom": {
        "name": "Custom OpenAI-Compatible",
        "base_url": "http://localhost:8000/v1",
        "default_model": "default",
        "models": [],
        "requires_key": False,
        "env_var": "CUSTOM_LLM_KEY",
    },
}


# Groq's recommended general-purpose model to fall back to when the configured
# model returns 404 (Groq decommissions models for non-enterprise tiers).
GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"


class LLMClient:
    """
    Unified LLM Client supporting OpenAI, Google Gemini, Groq, DeepSeek,
    Ollama, Anthropic Claude, Mistral AI, OpenRouter, and custom endpoints.
    """

    def __init__(
        self,
        api_key: str = "",
        *,
        provider: str = "openai",
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.provider = (provider or "openai").lower().strip()
        self.timeout = timeout

        preset = PROVIDER_PRESETS.get(self.provider, PROVIDER_PRESETS["openai"])
        self.base_url = (base_url or "").strip() or preset["base_url"]
        self.model = (model or "").strip() or preset["default_model"]

        # Handle API key requirements per provider
        resolved_key = (api_key or "").strip()
        if self.provider == "ollama":
            # Ollama doesn't require an API key
            self.api_key = resolved_key or "ollama"
        elif preset.get("requires_key", True) and not resolved_key:
            env_var = preset.get("env_var", "API_KEY")
            raise SQLGenerationError(
                f"{env_var} is required for provider '{preset['name']}'. "
                f"Set it in your environment, .env file, or enter it in settings."
            )
        else:
            self.api_key = resolved_key or "sk-dummy"

        # Initialize OpenAI client for all OpenAI-compatible providers
        if self.provider != "anthropic":
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
            )
        else:
            self._client = None

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

        user_content = "\n".join(user_parts)

        # Provider: Anthropic Claude via direct Messages API
        if self.provider == "anthropic":
            return self._generate_anthropic(user_content)

        # OpenAI and all OpenAI-compatible providers (Gemini, Groq, DeepSeek, Ollama, OpenRouter, Mistral)
        return self._generate_openai_compatible(user_content)

    def _generate_openai_compatible(self, user_content: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        except RateLimitError as exc:
            raise SQLGenerationError(
                f"{self.provider.upper()} rate limit exceeded. Wait and retry.", cause=exc
            ) from exc
        except APITimeoutError as exc:
            raise SQLGenerationError(
                f"{self.provider.upper()} request timed out.", cause=exc
            ) from exc
        except APIConnectionError as exc:
            raise SQLGenerationError(
                f"Could not reach {self.provider.upper()} API at {self.base_url}. Check network / host.",
                cause=exc,
            ) from exc
        except APIStatusError as exc:
            # Auto-recovery when the configured model no longer exists on Groq
            # (Groq decommissions models on short notice for non-enterprise tiers).
            if self.provider == "groq" and exc.status_code == 404 and self.model != GROQ_FALLBACK_MODEL:
                self.model = GROQ_FALLBACK_MODEL
                return self._generate_openai_compatible(user_content)

            raise SQLGenerationError(
                f"{self.provider.upper()} API error ({exc.status_code}): {exc.message}", cause=exc
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise SQLGenerationError(
                f"Unexpected {self.provider.upper()} LLM failure: {exc}", cause=exc
            ) from exc

        choice = response.choices[0].message.content if response.choices else None
        if not choice or not choice.strip():
            raise SQLGenerationError("LLM returned an empty completion.")

        return choice.strip()

    def generate_general_answer(self, question: str) -> str:
        """
        Answer a general-knowledge question directly from the LLM (no database).

        Used when the question cannot be answered from the connected schema.
        Returns a plain-markdown text answer.
        """
        system = (
            "You are a knowledgeable assistant. Answer the user's question clearly and "
            "concisely in plain text (no SQL).\n"
            "FORMAT RULES:\n"
            "1. If the question involves comparing or ranking entities by data "
            "(countries, companies, products, years, sports results, etc.), you MUST "
            "present the data as a markdown table. Table format: a header row like "
            "`| Rank | Country | GDP per capita (USD) | Year |`, then a separator line "
            "`|---|---|---|---|`, then one data row per entity. Show 5-10 rows when "
            "the question asks for top/bottom N or comparisons.\n"
            "2. Use bold (**text**) sparingly for key figures only.\n"
            "3. State the data year you are confident about and mention the source "
            "type (e.g. World Bank, government statistics).\n"
            "4. If unsure of exact figures, give best estimates marked with '~' and "
            "say so briefly rather than inventing precise numbers."
        )
        user = (
            "The user asked a question that cannot be answered from the connected "
            f"database. Answer it from general knowledge.\n\nQuestion: {question}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            detail = getattr(exc, "message", None) or str(exc)
            raise SQLGenerationError(f"{self.provider.upper()} error: {detail}", cause=exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise SQLGenerationError(f"Unexpected {self.provider.upper()} failure: {exc}", cause=exc) from exc

        choice = response.choices[0].message.content if response.choices else None
        if not choice or not choice.strip():
            raise SQLGenerationError("LLM returned an empty answer.")
        return choice.strip()

    def _generate_anthropic(self, user_content: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_content}],
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
                if res.status_code == 401:
                    raise SQLGenerationError("Anthropic authentication failed. Check ANTHROPIC_API_KEY.")
                if res.status_code == 429:
                    raise SQLGenerationError("Anthropic rate limit exceeded. Wait and retry.")
                if res.status_code != 200:
                    raise SQLGenerationError(f"Anthropic error ({res.status_code}): {res.text}")

                data = res.json()
                text_content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text_content += block.get("text", "")

                if not text_content.strip():
                    raise SQLGenerationError("Anthropic returned an empty completion.")
                return text_content.strip()
        except SQLGenerationError:
            raise
        except Exception as exc:
            raise SQLGenerationError(f"Anthropic request failed: {exc}", cause=exc) from exc

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
