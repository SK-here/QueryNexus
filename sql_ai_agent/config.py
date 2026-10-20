"""Runtime configuration loaded from environment and constructor args."""

import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider: openai, gemini, groq, deepseek, ollama, openrouter, mistral, anthropic, custom
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")

    # API keys (provider-specific or generic)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")

    # Model & Base URL
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_model: str = Field(default="", alias="LLM_MODEL")

    sql_url: str = Field(default="", alias="SQL_URL")
    sql_user: str = Field(default="", alias="SQL_USER")
    sql_password: str = Field(default="", alias="SQL_PASSWORD")

    # Safety: only allow SELECT by default
    allow_writes: bool = Field(default=False, alias="ALLOW_WRITES")
    max_rows: int = Field(default=100, alias="MAX_ROWS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    query_timeout_seconds: int = Field(default=30, alias="QUERY_TIMEOUT_SECONDS")

    def get_effective_api_key(self, provider: str | None = None) -> str:
        """Resolve API key based on provider or environment overrides."""
        p = (provider or self.llm_provider).lower()
        if self.llm_api_key:
            return self.llm_api_key
        if p == "openai":
            return self.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if p == "gemini":
            return (
                self.gemini_api_key
                or os.getenv("GEMINI_API_KEY", "")
                or os.getenv("GOOGLE_API_KEY", "")
                or self.openai_api_key
            )
        if p == "groq":
            return self.groq_api_key or os.getenv("GROQ_API_KEY", "")
        if p == "deepseek":
            return self.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if p == "anthropic":
            return self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if p == "openrouter":
            return self.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        if p == "mistral":
            return self.mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        if p == "ollama":
            return "ollama"
        return self.openai_api_key

    def get_effective_base_url(self, provider: str | None = None) -> str:
        """Resolve base URL without letting OpenAI default leak into other providers."""
        p = (provider or self.llm_provider).lower()
        if self.llm_base_url:
            return self.llm_base_url
        if p == "openai":
            return self.openai_base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        # For non-openai providers, do not leak OPENAI_BASE_URL
        return ""

    def get_effective_model(self, provider: str | None = None) -> str:
        """Resolve model without letting OpenAI default leak into other providers."""
        p = (provider or self.llm_provider).lower()
        if self.llm_model:
            return self.llm_model
        if p == "openai":
            return self.openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # For non-openai providers, do not leak OPENAI_MODEL
        return ""


def get_settings() -> Settings:
    return Settings()

