"""Runtime configuration loaded from environment and constructor args."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    sql_url: str = Field(default="", alias="SQL_URL")
    sql_user: str = Field(default="", alias="SQL_USER")
    sql_password: str = Field(default="", alias="SQL_PASSWORD")

    # Safety: only allow SELECT by default
    allow_writes: bool = Field(default=False, alias="ALLOW_WRITES")
    max_rows: int = Field(default=100, alias="MAX_ROWS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    query_timeout_seconds: int = Field(default=30, alias="QUERY_TIMEOUT_SECONDS")


def get_settings() -> Settings:
    return Settings()
