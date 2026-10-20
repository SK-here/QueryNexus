"""Web UI server for QueryNexus - SQL AI Agent."""

from __future__ import annotations

import csv
import io
import os
import time
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from sql_ai_agent.agent import AgentResponse, SQLAIAgent
from sql_ai_agent.config import Settings, get_settings
from sql_ai_agent.database import DatabaseManager
from sql_ai_agent.llm import LLMClient, PROVIDER_PRESETS
from sql_ai_agent.validator import validate_sql

app = FastAPI(title="QueryNexus Web UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectRequest(BaseModel):
    sql_url: str
    sql_user: str = "demo"
    sql_password: str = ""


class QueryRequest(BaseModel):
    question: str
    sql_url: str
    sql_user: str = "demo"
    sql_password: str = ""
    provider: str = "openai"
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    # Backward compatibility
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_base_url: Optional[str] = None
    allow_writes: bool = False
    execute: bool = True
    max_rows: int = 100
    allow_general: bool = True


class DirectSQLRequest(BaseModel):
    sql: str
    sql_url: str
    sql_user: str = "demo"
    sql_password: str = ""
    allow_writes: bool = False
    max_rows: int = 100


# Demo canned queries to allow instant preview even without an OpenAI API key
DEMO_OFFLINE_QUERIES: dict[str, str] = {
    "total paid revenue by country": (
        "SELECT c.country, SUM(o.amount) AS total_revenue\n"
        "FROM customers c\n"
        "JOIN orders o ON c.id = o.customer_id\n"
        "WHERE o.status = 'paid'\n"
        "GROUP BY c.country\n"
        "ORDER BY total_revenue DESC;"
    ),
    "top 5 customers by revenue": (
        "SELECT c.name, c.email, SUM(o.amount) AS total_spent\n"
        "FROM customers c\n"
        "JOIN orders o ON c.id = o.customer_id\n"
        "WHERE o.status = 'paid'\n"
        "GROUP BY c.id\n"
        "ORDER BY total_spent DESC\n"
        "LIMIT 5;"
    ),
    "list unpaid orders with customer details": (
        "SELECT o.id AS order_id, c.name, c.email, o.amount, o.status, o.ordered_at\n"
        "FROM orders o\n"
        "JOIN customers c ON o.customer_id = c.id\n"
        "WHERE o.status != 'paid'\n"
        "ORDER BY o.ordered_at DESC;"
    ),
    "count of orders per customer": (
        "SELECT c.name, COUNT(o.id) AS order_count, COALESCE(SUM(o.amount), 0) AS total_amount\n"
        "FROM customers c\n"
        "LEFT JOIN orders o ON c.id = o.customer_id\n"
        "GROUP BY c.id, c.name\n"
        "ORDER BY order_count DESC;"
    ),
}


@app.get("/api/config")
def get_config():
    settings = get_settings()
    detected_keys: dict[str, str] = {}

    for p, preset in PROVIDER_PRESETS.items():
        val = settings.get_effective_api_key(p)
        if val:
            detected_keys[p] = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"

    return {
        "providers": PROVIDER_PRESETS,
        "current_provider": settings.llm_provider,
        "detected_keys": detected_keys,
        "default_sql_url": settings.sql_url or "sqlite:///./demo.db",
        "default_sql_user": settings.sql_user or "demo",
        "default_sql_password": settings.sql_password or "",
        "allow_writes": settings.allow_writes,
        "max_rows": settings.max_rows,
    }


@app.post("/api/connect")
def test_connection(req: ConnectRequest):
    try:
        db = DatabaseManager(
            sql_url=req.sql_url,
            sql_user=req.sql_user,
            sql_password=req.sql_password,
            query_timeout_seconds=10,
        )
        db.connect()
        schema = db.get_schema()

        # Gather table names, column count, and estimate row counts
        tables_summary = []
        for table in schema.tables:
            row_count = 0
            try:
                res = db.execute(f"SELECT COUNT(*) AS c FROM {table.name}")
                if res.rows:
                    row_count = list(res.rows[0].values())[0]
            except Exception:
                row_count = -1

            tables_summary.append({
                "name": table.name,
                "columns": [
                    {
                        "name": col.name,
                        "type": col.type,
                        "nullable": col.nullable,
                        "primary_key": col.primary_key,
                    }
                    for col in table.columns
                ],
                "foreign_keys": table.foreign_keys,
                "row_count": row_count,
            })

        db.close()
        return {
            "success": True,
            "dialect": schema.dialect,
            "tables": tables_summary,
            "message": f"Successfully connected to {schema.dialect} database with {len(schema.tables)} tables.",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


@app.post("/api/table-preview")
def preview_table(req: ConnectRequest, table_name: str):
    try:
        # Validate table name to avoid injection
        clean_name = "".join(c for c in table_name if c.isalnum() or c in ("_", "-"))
        if not clean_name:
            raise HTTPException(status_code=400, detail="Invalid table name")

        db = DatabaseManager(
            sql_url=req.sql_url,
            sql_user=req.sql_user,
            sql_password=req.sql_password,
            max_rows=15,
        )
        db.connect()
        result = db.execute(f"SELECT * FROM {clean_name} LIMIT 15;")
        db.close()
        return {
            "success": True,
            "table": clean_name,
            "columns": result.columns,
            "rows": result.rows,
            "row_count": result.row_count,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


class ListModelsRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@app.post("/api/list-models")
def list_provider_models(req: ListModelsRequest):
    provider = req.provider.lower().strip()
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["openai"])
    env_settings = get_settings()
    api_key = (req.api_key or "").strip() or env_settings.get_effective_api_key(provider)
    base_url = (req.base_url or "").strip() or preset["base_url"]

    if provider == "ollama":
        api_key = api_key or "ollama"

    if not api_key and preset.get("requires_key", True):
        return {
            "success": False,
            "error": f"{preset['env_var']} is required to fetch models.",
            "fallback_models": preset.get("models", []),
        }

    if provider == "anthropic":
        return {"success": True, "models": preset.get("models", [])}

    try:
        clean_base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"}
        with httpx.Client(timeout=10.0) as client:
            res = client.get(f"{clean_base}/models", headers=headers)
            if res.status_code != 200:
                return {
                    "success": False,
                    "error": f"Provider returned HTTP {res.status_code}",
                    "fallback_models": preset.get("models", []),
                }
            data = res.json()
            models_list = []
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                models_list = [item["id"] for item in data["data"] if "id" in item]
            elif isinstance(data, dict) and "models" in data:
                models_list = [item.get("id") or item.get("name") for item in data["models"]]

            return {"success": True, "models": sorted(models_list)}
    except Exception as exc:
        return {"success": False, "error": str(exc), "fallback_models": preset.get("models", [])}


@app.post("/api/ask")
def ask_question(req: QueryRequest):
    start_time = time.time()
    question = req.question.strip()
    if not question:
        return {
            "success": False,
            "error": "Question cannot be empty.",
            "error_type": "EmptyQuestion",
        }

    provider = (req.provider or "openai").lower().strip()
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["openai"])

    # Resolve API key
    env_settings = get_settings()
    api_key = (
        (req.api_key or "").strip()
        or (req.openai_api_key or "").strip()
        or env_settings.get_effective_api_key(provider)
    )

    if provider == "openai":
        model = (
            (req.model or "").strip()
            or (req.openai_model or "").strip()
            or env_settings.openai_model
            or preset["default_model"]
        )
        base_url = (
            (req.base_url or "").strip()
            or (req.openai_base_url or "").strip()
            or env_settings.openai_base_url
            or preset["base_url"]
        )
    else:
        # Non-openai provider: ignore any stale gpt-* model or api.openai.com base_url
        raw_model = (req.model or "").strip() or env_settings.llm_model
        if not raw_model or raw_model.startswith("gpt-") or raw_model.startswith("o1-") or raw_model.startswith("o3-"):
            model = preset["default_model"]
        else:
            model = raw_model

        raw_url = (req.base_url or "").strip() or env_settings.llm_base_url
        if not raw_url or "api.openai.com" in raw_url:
            base_url = preset["base_url"]
        else:
            base_url = raw_url


    # Check if this matches a demo question and we have no API key
    normalized_q = question.lower().strip().rstrip("?.")
    if not api_key and preset.get("requires_key", True):
        matched_demo_sql = DEMO_OFFLINE_QUERIES.get(normalized_q)
        if matched_demo_sql:
            try:
                db = DatabaseManager(
                    sql_url=req.sql_url,
                    sql_user=req.sql_user,
                    sql_password=req.sql_password,
                    max_rows=req.max_rows,
                )
                db.connect()
                res = db.execute(matched_demo_sql)
                db.close()
                elapsed = round(time.time() - start_time, 3)
                return {
                    "success": True,
                    "question": question,
                    "sql": matched_demo_sql,
                    "columns": res.columns,
                    "rows": res.rows,
                    "row_count": res.row_count,
                    "truncated": res.truncated,
                    "retries_used": 0,
                    "execution_time": elapsed,
                    "is_demo_fallback": True,
                    "message": f"Demo query bank result. To run arbitrary questions with {preset['name']}, add your {preset['env_var']} in settings.",
                }
            except Exception as exc:
                return {
                    "success": False,
                    "question": question,
                    "sql": matched_demo_sql,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
        else:
            return {
                "success": False,
                "question": question,
                "error": f"{preset['env_var']} is required for {preset['name']}. Please open 'Connection & Keys' to enter your key or switch to Ollama for free local inference.",
                "error_type": "MissingAPIKey",
            }

    # Build agent with custom settings
    custom_settings = Settings(
        llm_provider=provider,
        llm_api_key=api_key,
        llm_model=model,
        llm_base_url=base_url,
        openai_api_key=api_key if provider == "openai" else "",
        sql_url=req.sql_url,
        sql_user=req.sql_user,
        sql_password=req.sql_password,
        allow_writes=req.allow_writes,
        max_rows=req.max_rows,
    )

    try:
        llm = LLMClient(
            api_key=api_key,
            provider=provider,
            model=model,
            base_url=base_url,
        )
        agent = SQLAIAgent(
            sql_url=req.sql_url,
            sql_user=req.sql_user,
            sql_password=req.sql_password,
            settings=custom_settings,
            allow_writes=req.allow_writes,
            max_rows=req.max_rows,
            llm=llm,
        )
        print(f"[/api/ask] Question: '{question}' | Provider: {provider} | Model: {model}")
        response: AgentResponse = agent.ask(question, execute=req.execute, allow_general=req.allow_general)
        agent.close()

        resp_dict = response.to_dict()
        resp_dict["execution_time"] = round(time.time() - start_time, 3)
        resp_dict["provider"] = provider
        resp_dict["model"] = getattr(llm, "model", model)

        if not response.success:
            print(f"[/api/ask FAILED] Type: {response.error_type} | Error: {response.error}")
        else:
            print(f"[/api/ask SUCCESS] Rows: {response.row_count} | SQL:\n{response.sql}")

        return resp_dict
    except Exception as exc:
        print(f"[/api/ask EXCEPTION] {type(exc).__name__}: {exc}")
        return {
            "success": False,
            "question": question,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "execution_time": round(time.time() - start_time, 3),
        }


@app.post("/api/execute-sql")
def execute_sql(req: DirectSQLRequest):
    start_time = time.time()
    raw_sql = req.sql.strip()
    if not raw_sql:
        return {"success": False, "error": "SQL statement cannot be empty."}

    try:
        validated = validate_sql(raw_sql, allow_writes=req.allow_writes)
        db = DatabaseManager(
            sql_url=req.sql_url,
            sql_user=req.sql_user,
            sql_password=req.sql_password,
            max_rows=req.max_rows,
        )
        db.connect()
        res = db.execute(validated)
        db.close()
        return {
            "success": True,
            "sql": validated,
            "columns": res.columns,
            "rows": res.rows,
            "row_count": res.row_count,
            "truncated": res.truncated,
            "execution_time": round(time.time() - start_time, 3),
        }
    except Exception as exc:
        return {
            "success": False,
            "sql": raw_sql,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "execution_time": round(time.time() - start_time, 3),
        }


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QueryNexus — Multi-Model SQL AI Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/sql.min.js"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#eef2ff',
              100: '#e0e7ff',
              400: '#818cf8',
              500: '#6366f1',
              600: '#4f46e5',
              700: '#4338ca',
            }
          }
        }
      }
    }
  </script>
  <style>
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans antialiased selection:bg-brand-500 selection:text-white">

  <!-- Header -->
  <header class="border-b border-slate-800/80 bg-slate-900/70 backdrop-blur sticky top-0 z-30 px-6 py-3 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
        <i data-lucide="database" class="w-5 h-5 text-white"></i>
      </div>
      <div>
        <div class="flex items-center gap-2">
          <span class="text-lg font-bold tracking-tight text-white">Query<span class="text-indigo-400">Nexus</span></span>
          <span class="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">Multi-LLM Studio</span>
        </div>
        <p class="text-xs text-slate-400">Natural-Language to SQL across OpenAI, Gemini, Groq, DeepSeek, Claude, Ollama</p>
      </div>
    </div>

    <div class="flex items-center gap-3">
      <!-- Active Provider Badge -->
      <button onclick="toggleConfigModal(true)" id="providerPill" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-800 bg-slate-900/80 hover:bg-slate-800 text-xs text-indigo-300 transition">
        <i data-lucide="cpu" class="w-3.5 h-3.5 text-indigo-400"></i>
        <span id="activeModelLabel">OpenAI (gpt-4o-mini)</span>
      </button>

      <!-- Database Connection Status Badge -->
      <div id="connectionStatusBadge" class="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-800 bg-slate-900 text-xs">
        <span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse" id="statusDot"></span>
        <span class="text-slate-300 font-medium" id="statusText">Checking database...</span>
      </div>

      <!-- Settings / Connection Button -->
      <button onclick="toggleConfigModal(true)" class="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-700 hover:border-slate-600 bg-slate-800/60 hover:bg-slate-800 text-xs text-slate-200 transition">
        <i data-lucide="settings" class="w-4 h-4"></i>
        <span>Settings</span>
      </button>
    </div>
  </header>

  <!-- Main Container -->
  <div class="flex-1 flex overflow-hidden">

    <!-- Left Sidebar: Schema Explorer -->
    <aside class="w-80 border-r border-slate-800/80 bg-slate-900/30 flex flex-col shrink-0">
      <div class="p-4 border-b border-slate-800/80 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <i data-lucide="table" class="w-4 h-4 text-indigo-400"></i>
          <span class="text-xs font-bold uppercase tracking-wider text-slate-300">Schema Explorer</span>
        </div>
        <button onclick="refreshSchema()" title="Refresh Schema" class="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition">
          <i data-lucide="refresh-cw" class="w-3.5 h-3.5"></i>
        </button>
      </div>

      <div id="schemaContainer" class="flex-1 overflow-y-auto p-3 space-y-2">
        <div class="p-4 text-center text-xs text-slate-500">
          <i data-lucide="loader-2" class="w-5 h-5 mx-auto animate-spin mb-2 text-indigo-400"></i>
          Loading database schema...
        </div>
      </div>

      <!-- Footer info -->
      <div class="p-3 border-t border-slate-800/80 text-[11px] text-slate-500 flex items-center justify-between">
        <span id="schemaDialect">Dialect: SQLite</span>
        <span id="schemaTableCount">0 Tables</span>
      </div>
    </aside>

    <!-- Main Content Area -->
    <main class="flex-1 flex flex-col overflow-y-auto">

      <!-- Query Mode Switcher -->
      <div class="px-6 pt-5 pb-2 border-b border-slate-800/80 flex items-center justify-between">
        <div class="flex items-center gap-2 border-b-2 border-transparent">
          <button id="tabAiBtn" onclick="switchTab('ai')" class="px-4 py-2 text-sm font-semibold text-indigo-400 border-b-2 border-indigo-500 flex items-center gap-2">
            <i data-lucide="sparkles" class="w-4 h-4"></i>
            Ask AI in Natural Language
          </button>
          <button id="tabSqlBtn" onclick="switchTab('sql')" class="px-4 py-2 text-sm font-semibold text-slate-400 hover:text-slate-200 border-b-2 border-transparent flex items-center gap-2">
            <i data-lucide="terminal" class="w-4 h-4"></i>
            Direct SQL Runner
          </button>
        </div>

        <div class="flex items-center gap-4 text-xs text-slate-400">
          <label class="flex items-center gap-2 cursor-pointer select-none">
            <input type="checkbox" id="allowWritesCheck" class="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-0">
            <span>Allow Writes (DML/DDL)</span>
          </label>
          <label class="flex items-center gap-2 cursor-pointer select-none" title="If a question can't be answered from the database, let the model answer from its own knowledge">
            <input type="checkbox" id="allowGeneralCheck" class="rounded bg-slate-800 border-slate-700 text-emerald-600 focus:ring-0" checked>
            <span>Allow General Knowledge Answers</span>
          </label>
        </div>
      </div>

      <div class="p-6 space-y-6 flex-1">

        <!-- AI Assistant Panel -->
        <div id="aiPanel" class="space-y-4">
          <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 shadow-xl">
            <div class="flex items-center justify-between mb-2">
              <label for="aiQuestion" class="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <i data-lucide="message-square-quote" class="w-3.5 h-3.5 text-indigo-400"></i>
                Ask question in plain English
              </label>
              <span class="text-[11px] text-slate-500">Press <kbd class="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Ctrl + Enter</kbd> to execute</span>
            </div>

            <div class="relative">
              <textarea id="aiQuestion" rows="3" placeholder="e.g. Total paid revenue by country, or Top 5 customers by revenue..."
                class="w-full bg-slate-950/80 border border-slate-700/80 focus:border-indigo-500 rounded-lg p-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 resize-none font-sans"></textarea>
            </div>

            <div class="mt-3 flex flex-wrap items-center justify-between gap-3">
              <!-- Example Prompt Chips -->
              <div class="flex flex-wrap items-center gap-2">
                <span class="text-xs text-slate-500 font-medium">Try:</span>
                <button onclick="setQuery('total paid revenue by country')" class="text-xs bg-slate-800/80 hover:bg-slate-800 text-slate-300 hover:text-indigo-300 border border-slate-700/60 rounded-full px-3 py-1 transition">
                  Total revenue by country
                </button>
                <button onclick="setQuery('top 5 customers by revenue')" class="text-xs bg-slate-800/80 hover:bg-slate-800 text-slate-300 hover:text-indigo-300 border border-slate-700/60 rounded-full px-3 py-1 transition">
                  Top 5 customers
                </button>
                <button onclick="setQuery('list unpaid orders with customer details')" class="text-xs bg-slate-800/80 hover:bg-slate-800 text-slate-300 hover:text-indigo-300 border border-slate-700/60 rounded-full px-3 py-1 transition">
                  Unpaid orders
                </button>
                <button onclick="setQuery('count of orders per customer')" class="text-xs bg-slate-800/80 hover:bg-slate-800 text-slate-300 hover:text-indigo-300 border border-slate-700/60 rounded-full px-3 py-1 transition">
                  Orders per customer
                </button>
              </div>

              <!-- Action buttons -->
              <div class="flex items-center gap-2 ml-auto">
                <button id="btnGenerateOnly" onclick="runAiQuery(false)" class="px-3 py-2 rounded-lg border border-slate-700 hover:border-slate-600 bg-slate-800 hover:bg-slate-750 text-xs text-slate-300 font-medium transition flex items-center gap-1.5">
                  <i data-lucide="code" class="w-3.5 h-3.5"></i>
                  Generate SQL Only
                </button>
                <button id="btnAskAi" onclick="runAiQuery(true)" class="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition flex items-center gap-2">
                  <i data-lucide="sparkles" class="w-3.5 h-3.5"></i>
                  Ask & Execute
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Direct SQL Panel -->
        <div id="sqlPanel" class="space-y-4 hidden">
          <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 shadow-xl">
            <div class="flex items-center justify-between mb-2">
              <label for="rawSqlInput" class="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <i data-lucide="terminal" class="w-3.5 h-3.5 text-indigo-400"></i>
                Direct SQL Query Editor
              </label>
              <span class="text-[11px] text-slate-500">Read-only enforced unless 'Allow Writes' is checked</span>
            </div>

            <textarea id="rawSqlInput" rows="4" placeholder="SELECT * FROM customers LIMIT 10;"
              class="w-full font-mono text-xs bg-slate-950/80 border border-slate-700/80 focus:border-indigo-500 rounded-lg p-3 text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500 resize-none"></textarea>

            <div class="mt-3 flex items-center justify-end gap-2">
              <button onclick="runDirectSql()" class="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition flex items-center gap-2">
                <i data-lucide="play" class="w-3.5 h-3.5 fill-current"></i>
                Run SQL
              </button>
            </div>
          </div>
        </div>

        <!-- Result Section -->
        <div id="resultsCard" class="bg-slate-900/40 border border-slate-800/80 rounded-xl overflow-hidden shadow-xl">
          <!-- Result Header -->
          <div class="p-4 bg-slate-900/70 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
            <div class="flex items-center gap-3">
              <span class="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <i data-lucide="bar-chart-2" class="w-4 h-4 text-indigo-400"></i>
                Query Results
              </span>
              <span id="resultBadge" class="hidden text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-800 text-slate-400"></span>
              <span id="providerModelBadge" class="hidden text-[11px] font-mono px-2 py-0.5 rounded bg-indigo-950/70 text-indigo-300 border border-indigo-800/40"></span>
              <span id="executionTimeBadge" class="hidden text-[11px] text-slate-500 font-mono"></span>
            </div>

            <div class="flex items-center gap-2">
              <button id="btnCopySql" onclick="copySql()" class="hidden px-2.5 py-1 rounded border border-slate-700 bg-slate-800 hover:bg-slate-750 text-xs text-slate-300 flex items-center gap-1.5 transition">
                <i data-lucide="copy" class="w-3.5 h-3.5"></i>
                <span>Copy SQL</span>
              </button>
              <button id="btnExportCsv" onclick="exportCsv()" class="hidden px-2.5 py-1 rounded border border-slate-700 bg-slate-800 hover:bg-slate-750 text-xs text-slate-300 flex items-center gap-1.5 transition">
                <i data-lucide="download" class="w-3.5 h-3.5"></i>
                <span>Export CSV</span>
              </button>
            </div>
          </div>

          <!-- Generated SQL View -->
          <div id="sqlViewContainer" class="hidden border-b border-slate-800/80 bg-slate-950 p-4">
            <div class="flex items-center justify-between text-xs text-slate-400 mb-2">
              <span class="font-semibold text-indigo-400 flex items-center gap-1.5">
                <i data-lucide="code-2" class="w-3.5 h-3.5"></i>
                Generated SQL
              </span>
              <span id="retriesUsedTag" class="text-[11px] text-slate-500"></span>
            </div>
            <pre class="p-3 bg-slate-900/90 rounded-lg text-xs font-mono overflow-x-auto border border-slate-800"><code id="sqlCodeBlock" class="language-sql"></code></pre>
          </div>

          <!-- General-knowledge answer view -->
          <div id="generalAnswerContainer" class="hidden border-b border-slate-800/80 bg-slate-950 p-4">
            <div class="flex items-center gap-1.5 text-xs font-semibold text-emerald-400 mb-2">
              <i data-lucide="lightbulb" class="w-3.5 h-3.5"></i>
              General Knowledge Answer
            </div>
            <div id="generalAnswerText" class="text-xs text-slate-200 leading-relaxed general-md"></div>
          </div>

          <!-- Demo Query Notice Banner -->
          <div id="demoNoticeBanner" class="hidden p-3 bg-indigo-950/40 border-b border-indigo-800/40 text-xs text-indigo-300 flex items-center justify-between">
            <div class="flex items-center gap-2">
              <i data-lucide="info" class="w-4 h-4 shrink-0 text-indigo-400"></i>
              <span id="demoNoticeText">Loaded from demo queries. Add your LLM API key in settings or switch to Ollama for local generation.</span>
            </div>
            <button onclick="toggleConfigModal(true)" class="underline font-medium hover:text-white shrink-0">Open Settings</button>
          </div>

          <!-- Alert / Clarification / Error Banner -->
          <div id="alertBanner" class="hidden p-4 border-b border-slate-800/80"></div>

          <!-- Results Table Body -->
          <div id="tableContainer" class="p-6">
            <div id="tableEmptyState" class="py-12 text-center text-slate-500 space-y-3">
              <div class="w-12 h-12 mx-auto rounded-full bg-slate-800/60 border border-slate-700/60 flex items-center justify-center text-slate-400">
                <i data-lucide="database" class="w-6 h-6"></i>
              </div>
              <p class="text-sm">Ready to explore. Ask a question above or pick a demo example to preview results.</p>
            </div>

            <div id="tableWrapper" class="hidden overflow-x-auto rounded-lg border border-slate-800">
              <table id="dataTable" class="w-full text-left text-xs border-collapse">
                <thead id="dataThead" class="bg-slate-900 text-slate-300 uppercase tracking-wider font-semibold border-b border-slate-800"></thead>
                <tbody id="dataTbody" class="divide-y divide-slate-800/60 font-mono text-slate-300"></tbody>
              </table>
            </div>
          </div>
        </div>

      </div>
    </main>
  </div>

  <!-- Table Preview Modal -->
  <div id="tablePreviewModal" class="hidden fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-800 rounded-xl w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl">
      <div class="p-4 border-b border-slate-800 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <i data-lucide="table" class="w-4 h-4 text-indigo-400"></i>
          <span id="previewModalTitle" class="text-sm font-bold text-white">Table Preview</span>
        </div>
        <button onclick="closeTablePreviewModal()" class="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800">
          <i data-lucide="x" class="w-4 h-4"></i>
        </button>
      </div>
      <div id="previewModalContent" class="p-4 overflow-auto flex-1"></div>
    </div>
  </div>

  <!-- Settings / Multi-LLM Config Modal -->
  <div id="configModal" class="hidden fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-800 rounded-xl w-full max-w-lg shadow-2xl overflow-hidden max-h-[92vh] flex flex-col">
      <div class="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/80">
        <div class="flex items-center gap-2">
          <i data-lucide="settings" class="w-4 h-4 text-indigo-400"></i>
          <span class="text-sm font-bold text-white">Settings: LLM Provider & Database</span>
        </div>
        <button onclick="toggleConfigModal(false)" class="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800">
          <i data-lucide="x" class="w-4 h-4"></i>
        </button>
      </div>

      <div class="p-5 space-y-4 text-xs overflow-y-auto flex-1">
        
        <!-- LLM Provider Selection -->
        <div class="p-3.5 rounded-lg border border-indigo-900/50 bg-indigo-950/20 space-y-3">
          <div class="flex items-center justify-between">
            <label class="text-xs font-bold text-indigo-300 flex items-center gap-1.5">
              <i data-lucide="cpu" class="w-4 h-4 text-indigo-400"></i>
              AI / LLM Provider
            </label>
            <span id="providerKeyIndicator" class="text-[10px] text-slate-400 font-mono"></span>
          </div>

          <select id="cfgProvider" onchange="onProviderChanged()" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 font-medium">
            <option value="openai">OpenAI (GPT-4o, GPT-4o-mini)</option>
            <option value="gemini">Google Gemini (Gemini 2.0 Flash, 1.5 Pro)</option>
            <option value="groq">Groq Cloud (Llama 3.3 70B - Ultra Fast)</option>
            <option value="deepseek">DeepSeek (DeepSeek V3, R1)</option>
            <option value="ollama">Ollama (Local / Free / No API Key)</option>
            <option value="anthropic">Anthropic Claude (Claude 3.5 Sonnet)</option>
            <option value="openrouter">OpenRouter (Universal Model Gateway)</option>
            <option value="mistral">Mistral AI (Codestral, Mistral Small)</option>
            <option value="custom">Custom (OpenAI-Compatible / vLLM / LM Studio)</option>
          </select>

          <!-- Model & Base URL Inputs -->
          <div class="grid grid-cols-2 gap-2">
            <div>
              <div class="flex items-center justify-between mb-1">
                <label for="cfgModel" class="block text-slate-400 font-medium">Model Name</label>
                <button type="button" onclick="fetchLiveModels()" id="btnFetchModels" class="text-[10px] text-indigo-400 hover:text-indigo-300 flex items-center gap-1 underline">
                  <i data-lucide="refresh-cw" class="w-3 h-3"></i>
                  <span>Live Models</span>
                </button>
              </div>
              <input type="text" id="cfgModel" list="modelsDatalist" placeholder="llama-3.1-8b-instant" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono text-[11px]">
              <datalist id="modelsDatalist"></datalist>
            </div>
            <div>
              <div class="flex items-center justify-between mb-1">
                <label for="cfgBaseUrl" class="block text-slate-400 font-medium">Base Endpoint URL</label>
                <button type="button" onclick="resetEndpointToDefault()" class="text-[10px] text-indigo-400 hover:text-indigo-300 underline">Reset to Default</button>
              </div>
              <input type="text" id="cfgBaseUrl" placeholder="https://api.openai.com/v1" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono text-[11px]">
            </div>
          </div>

          <!-- Quick Model Selection Chips -->
          <div>
            <div class="flex items-center justify-between text-[10px] text-slate-500 mb-1">
              <span>Quick Select / Verified Models:</span>
              <span id="modelsCountBadge" class="text-[9px] text-slate-400"></span>
            </div>
            <div id="modelChipsContainer" class="flex flex-wrap gap-1.5"></div>
          </div>

          <!-- API Key Input -->
          <div>
            <label for="cfgApiKey" class="block text-slate-400 font-medium mb-1 flex items-center justify-between">
              <span id="apiKeyLabel">API Key</span>
              <span id="apiKeyHelp" class="text-[10px] text-slate-500"></span>
            </label>
            <input type="password" id="cfgApiKey" placeholder="Enter API Key" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono">
          </div>
        </div>

        <!-- Database Connection -->
        <div class="space-y-3 pt-2">
          <label class="block text-slate-400 font-bold uppercase tracking-wider text-[11px]">Database Connection</label>
          
          <!-- Presets -->
          <div class="flex gap-2">
            <button onclick="applyPreset('sqlite')" class="flex-1 py-1.5 px-2 rounded border border-slate-700 bg-slate-800/80 hover:bg-slate-800 text-slate-200 text-center font-medium transition">
              Demo SQLite
            </button>
            <button onclick="applyPreset('postgres')" class="flex-1 py-1.5 px-2 rounded border border-slate-700 bg-slate-800/80 hover:bg-slate-800 text-slate-200 text-center font-medium transition">
              PostgreSQL
            </button>
            <button onclick="applyPreset('mysql')" class="flex-1 py-1.5 px-2 rounded border border-slate-700 bg-slate-800/80 hover:bg-slate-800 text-slate-200 text-center font-medium transition">
              MySQL
            </button>
          </div>

          <div>
            <label for="cfgSqlUrl" class="block text-slate-400 font-medium mb-1">Database SQL URL</label>
            <input type="text" id="cfgSqlUrl" placeholder="sqlite:///./demo.db" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono">
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label for="cfgSqlUser" class="block text-slate-400 font-medium mb-1">Database User</label>
              <input type="text" id="cfgSqlUser" placeholder="demo" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>
            <div>
              <label for="cfgSqlPassword" class="block text-slate-400 font-medium mb-1">Database Password</label>
              <input type="password" id="cfgSqlPassword" placeholder="(empty for SQLite)" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>
          </div>
        </div>

        <div id="cfgModalError" class="hidden p-2.5 rounded bg-rose-950/50 border border-rose-800/60 text-rose-300"></div>
      </div>

      <div class="p-4 bg-slate-900/80 border-t border-slate-800 flex items-center justify-between">
        <button onclick="testAndConnect()" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition flex items-center gap-2">
          <i data-lucide="plug" class="w-3.5 h-3.5"></i>
          Test Database
        </button>
        <button onclick="saveConfigModal()" class="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition">
          Save Settings
        </button>
      </div>
    </div>
  </div>

  <script>
    // App State
    let configData = null;

    const state = {
      provider: localStorage.getItem('qn_provider') || 'openai',
      apiKey: localStorage.getItem('qn_api_key') || '',
      model: localStorage.getItem('qn_model') || 'gpt-4o-mini',
      baseUrl: localStorage.getItem('qn_base_url') || 'https://api.openai.com/v1',
      sqlUrl: localStorage.getItem('qn_sql_url') || 'sqlite:///./demo.db',
      sqlUser: localStorage.getItem('qn_sql_user') || 'demo',
      sqlPassword: localStorage.getItem('qn_sql_password') || '',
      schema: null,
      currentResults: null,
      currentSql: null,
      activeTab: 'ai'
    };

    document.addEventListener('DOMContentLoaded', async () => {
      lucide.createIcons();
      initKeyboard();
      await fetchInitialConfig();
      updateHeaderBadge();
      await refreshSchema();
    });

    function initKeyboard() {
      const aiInput = document.getElementById('aiQuestion');
      aiInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
          e.preventDefault();
          runAiQuery(true);
        }
      });
    }

    async function fetchInitialConfig() {
      try {
        const res = await fetch('/api/config');
        configData = await res.json();
        
        // Remove legacy contaminated keys
        localStorage.removeItem('qn_base_url');
        localStorage.removeItem('qn_model');

        if (!localStorage.getItem('qn_provider') && configData.current_provider) {
          state.provider = configData.current_provider;
        }
        if (!state.sqlUrl && configData.default_sql_url) state.sqlUrl = configData.default_sql_url;

        const preset = configData.providers[state.provider];
        if (preset) {
          state.model = localStorage.getItem('qn_model_' + state.provider) || preset.default_model;
          // Auto-migrate Groq models decommissioned for free/developer tiers (2026-08-16)
          const GROQ_LEGACY_MODELS = ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it', 'deepseek-r1-distill-llama-70b', 'llama3-70b-8192', 'llama3-8b-8192'];
          if (state.provider === 'groq' && (!state.model || GROQ_LEGACY_MODELS.includes(state.model))) {
            state.model = preset.default_model;
            localStorage.setItem('qn_model_groq', state.model);
          }
          state.baseUrl = localStorage.getItem('qn_url_' + state.provider) || preset.base_url;
          state.apiKey = localStorage.getItem('qn_key_' + state.provider) || localStorage.getItem('qn_api_key') || '';
        }
      } catch (e) {
        console.error('Failed to load initial config:', e);
      }
    }

    function updateHeaderBadge() {
      const pill = document.getElementById('activeModelLabel');
      if (pill) {
        const pName = configData?.providers?.[state.provider]?.name || state.provider;
        pill.textContent = `${pName} (${state.model})`;
      }
    }

    function onProviderChanged() {
      const select = document.getElementById('cfgProvider');
      const p = select.value;
      const preset = configData?.providers?.[p];
      if (!preset) return;

      const savedModel = localStorage.getItem('qn_model_' + p);
      const savedUrl = localStorage.getItem('qn_url_' + p);
      const savedKey = localStorage.getItem('qn_key_' + p);

      document.getElementById('cfgModel').value = savedModel || preset.default_model;
      document.getElementById('cfgBaseUrl').value = savedUrl || preset.base_url;

      const keyLabel = document.getElementById('apiKeyLabel');
      const keyHelp = document.getElementById('apiKeyHelp');
      const keyInput = document.getElementById('cfgApiKey');

      keyInput.value = savedKey || (p === state.provider ? state.apiKey : '');
      keyLabel.textContent = preset.env_var || 'API Key';

      if (p === 'ollama') {
        keyHelp.textContent = 'Not required for local Ollama';
        keyInput.placeholder = '(No key required)';
      } else if (p === 'gemini') {
        keyHelp.textContent = 'Google AI Studio Key (AIzaSy...)';
        keyInput.placeholder = 'AIzaSy...';
      } else if (p === 'anthropic') {
        keyHelp.textContent = 'Anthropic Key (sk-ant-...)';
        keyInput.placeholder = 'sk-ant-...';
      } else if (p === 'groq') {
        keyHelp.textContent = 'Groq Cloud Key (gsk_...)';
        keyInput.placeholder = 'gsk_...';
      } else if (p === 'deepseek') {
        keyHelp.textContent = 'DeepSeek Key (sk-...)';
        keyInput.placeholder = 'sk-...';
      } else {
        keyHelp.textContent = 'Required for generation';
        keyInput.placeholder = 'sk-...';
      }

      const detected = configData?.detected_keys?.[p];
      const indicator = document.getElementById('providerKeyIndicator');
      if (detected) {
        indicator.textContent = 'Key detected: ' + detected;
        indicator.className = 'text-[10px] text-emerald-400 font-mono';
      } else if (p === 'ollama') {
        indicator.textContent = 'Local runner';
        indicator.className = 'text-[10px] text-indigo-400 font-mono';
      } else {
        indicator.textContent = 'No key set in .env';
        indicator.className = 'text-[10px] text-slate-500 font-mono';
      }

      renderModelChips(preset.models || []);
    }

    function renderModelChips(models) {
      const container = document.getElementById('modelChipsContainer');
      const datalist = document.getElementById('modelsDatalist');
      const badge = document.getElementById('modelsCountBadge');
      if (!container) return;

      container.innerHTML = '';
      if (datalist) datalist.innerHTML = '';

      if (!models || models.length === 0) {
        container.innerHTML = '<span class="text-[10px] text-slate-500 italic">No presets defined. Type any model name above.</span>';
        if (badge) badge.textContent = '';
        return;
      }

      if (badge) badge.textContent = `(${models.length} available)`;

      if (datalist) {
        models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m;
          datalist.appendChild(opt);
        });
      }

      const chipsToShow = models.slice(0, 8);
      chipsToShow.forEach(m => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'text-[10px] px-2 py-0.5 rounded-full border border-slate-700/80 bg-slate-800/80 hover:bg-slate-750 text-slate-300 hover:text-indigo-300 hover:border-indigo-500/50 transition font-mono truncate max-w-[200px]';
        chip.title = m;
        chip.textContent = m;
        chip.onclick = () => {
          document.getElementById('cfgModel').value = m;
        };
        container.appendChild(chip);
      });
    }

    async function fetchLiveModels() {
      const btn = document.getElementById('btnFetchModels');
      const provider = document.getElementById('cfgProvider').value;
      const apiKey = document.getElementById('cfgApiKey').value.trim();
      const baseUrl = document.getElementById('cfgBaseUrl').value.trim();

      const origHtml = btn.innerHTML;
      btn.innerHTML = '<i data-lucide="loader-2" class="w-3 h-3 animate-spin"></i> Loading...';
      lucide.createIcons();

      try {
        const res = await fetch('/api/list-models', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            provider: provider,
            api_key: apiKey,
            base_url: baseUrl,
          })
        });
        const data = await res.json();
        if (data.success && data.models && data.models.length > 0) {
          renderModelChips(data.models);
          alert(`Successfully loaded ${data.models.length} model(s) from ${provider}!`);
        } else {
          alert('Could not fetch model list: ' + (data.error || 'Unknown error'));
          if (data.fallback_models) renderModelChips(data.fallback_models);
        }
      } catch (err) {
        alert('Failed to connect to provider: ' + err.message);
      } finally {
        btn.innerHTML = origHtml;
        lucide.createIcons();
      }
    }

    function resetEndpointToDefault() {
      const select = document.getElementById('cfgProvider');
      const p = select.value;
      const preset = configData?.providers?.[p];
      if (!preset) return;
      document.getElementById('cfgBaseUrl').value = preset.base_url;
      document.getElementById('cfgModel').value = preset.default_model;
      localStorage.removeItem('qn_url_' + p);
      localStorage.removeItem('qn_model_' + p);
      renderModelChips(preset.models || []);
    }

    function switchTab(tab) {
      state.activeTab = tab;
      const aiPanel = document.getElementById('aiPanel');
      const sqlPanel = document.getElementById('sqlPanel');
      const tabAiBtn = document.getElementById('tabAiBtn');
      const tabSqlBtn = document.getElementById('tabSqlBtn');

      if (tab === 'ai') {
        aiPanel.classList.remove('hidden');
        sqlPanel.classList.add('hidden');
        tabAiBtn.className = 'px-4 py-2 text-sm font-semibold text-indigo-400 border-b-2 border-indigo-500 flex items-center gap-2';
        tabSqlBtn.className = 'px-4 py-2 text-sm font-semibold text-slate-400 hover:text-slate-200 border-b-2 border-transparent flex items-center gap-2';
      } else {
        aiPanel.classList.add('hidden');
        sqlPanel.classList.remove('hidden');
        tabSqlBtn.className = 'px-4 py-2 text-sm font-semibold text-indigo-400 border-b-2 border-indigo-500 flex items-center gap-2';
        tabAiBtn.className = 'px-4 py-2 text-sm font-semibold text-slate-400 hover:text-slate-200 border-b-2 border-transparent flex items-center gap-2';
      }
      lucide.createIcons();
    }

    function setQuery(text) {
      document.getElementById('aiQuestion').value = text;
      document.getElementById('aiQuestion').focus();
    }

    async function refreshSchema() {
      const container = document.getElementById('schemaContainer');
      const statusDot = document.getElementById('statusDot');
      const statusText = document.getElementById('statusText');

      container.innerHTML = `
        <div class="p-4 text-center text-xs text-slate-500">
          <i data-lucide="loader-2" class="w-5 h-5 mx-auto animate-spin mb-2 text-indigo-400"></i>
          Connecting to database...
        </div>
      `;
      lucide.createIcons();

      try {
        const res = await fetch('/api/connect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sql_url: state.sqlUrl,
            sql_user: state.sqlUser,
            sql_password: state.sqlPassword,
          })
        });
        const data = await res.json();

        if (data.success) {
          state.schema = data;
          statusDot.className = 'w-2 h-2 rounded-full bg-emerald-400';
          statusText.textContent = `${data.dialect} connected`;
          document.getElementById('schemaDialect').textContent = `Dialect: ${data.dialect}`;
          document.getElementById('schemaTableCount').textContent = `${data.tables.length} Tables`;

          if (data.tables.length === 0) {
            container.innerHTML = `<div class="p-4 text-center text-xs text-slate-500">No tables found.</div>`;
            return;
          }

          let html = '';
          data.tables.forEach(t => {
            const rowCountLabel = t.row_count >= 0 ? `${t.row_count} rows` : '';
            html += `
              <div class="border border-slate-800/80 rounded-lg bg-slate-900/60 overflow-hidden text-xs">
                <div class="p-2.5 flex items-center justify-between bg-slate-850 hover:bg-slate-800/50 cursor-pointer select-none" onclick="toggleTableDetails('${t.name}')">
                  <div class="flex items-center gap-2 overflow-hidden">
                    <i data-lucide="chevron-right" id="arrow-${t.name}" class="w-3.5 h-3.5 text-slate-400 transition-transform"></i>
                    <span class="font-bold text-slate-200 truncate">${t.name}</span>
                  </div>
                  <div class="flex items-center gap-1.5 shrink-0">
                    <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">${rowCountLabel}</span>
                    <button onclick="previewTable(event, '${t.name}')" title="Preview Rows" class="p-1 hover:text-indigo-400 text-slate-400 rounded hover:bg-slate-700/60 transition">
                      <i data-lucide="eye" class="w-3.5 h-3.5"></i>
                    </button>
                  </div>
                </div>
                <div id="cols-${t.name}" class="hidden p-2.5 border-t border-slate-800/60 bg-slate-950/40 space-y-1.5 text-[11px] font-mono">
                  ${t.columns.map(c => `
                    <div class="flex items-center justify-between text-slate-400">
                      <span class="flex items-center gap-1.5 text-slate-300">
                        ${c.primary_key ? '<span class="text-[9px] px-1 bg-amber-500/20 text-amber-300 rounded font-sans font-bold">PK</span>' : ''}
                        ${c.name}
                      </span>
                      <span class="text-slate-500 text-[10px]">${c.type}</span>
                    </div>
                  `).join('')}
                </div>
              </div>
            `;
          });
          container.innerHTML = html;
          lucide.createIcons();
        } else {
          statusDot.className = 'w-2 h-2 rounded-full bg-rose-400';
          statusText.textContent = 'Connection failed';
          container.innerHTML = `
            <div class="p-3 rounded bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300">
              <p class="font-bold mb-1">Failed to connect</p>
              <p class="text-[11px] break-all">${data.error || 'Check database credentials'}</p>
            </div>
          `;
        }
      } catch (err) {
        statusDot.className = 'w-2 h-2 rounded-full bg-rose-400';
        statusText.textContent = 'Server offline';
        container.innerHTML = `<div class="p-4 text-center text-xs text-rose-400">Error connecting to local server.</div>`;
      }
    }

    function toggleTableDetails(tableName) {
      const el = document.getElementById(`cols-${tableName}`);
      const arrow = document.getElementById(`arrow-${tableName}`);
      if (el.classList.contains('hidden')) {
        el.classList.remove('hidden');
        arrow.classList.add('rotate-90');
      } else {
        el.classList.add('hidden');
        arrow.classList.remove('rotate-90');
      }
    }

    async function previewTable(event, tableName) {
      event.stopPropagation();
      const modal = document.getElementById('tablePreviewModal');
      const title = document.getElementById('previewModalTitle');
      const content = document.getElementById('previewModalContent');

      title.textContent = `Table: ${tableName} (First 15 rows)`;
      content.innerHTML = `<div class="py-8 text-center text-xs text-slate-400"><i data-lucide="loader-2" class="w-5 h-5 mx-auto animate-spin mb-2 text-indigo-400"></i> Loading...</div>`;
      modal.classList.remove('hidden');
      lucide.createIcons();

      try {
        const res = await fetch(`/api/table-preview?table_name=${encodeURIComponent(tableName)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sql_url: state.sqlUrl,
            sql_user: state.sqlUser,
            sql_password: state.sqlPassword,
          })
        });
        const data = await res.json();
        if (data.success && data.rows.length > 0) {
          content.innerHTML = renderTableHtml(data.columns, data.rows);
        } else if (data.success && data.rows.length === 0) {
          content.innerHTML = `<div class="py-6 text-center text-xs text-slate-500">Table is empty.</div>`;
        } else {
          content.innerHTML = `<div class="p-3 text-xs text-rose-400">${data.error}</div>`;
        }
      } catch (e) {
        content.innerHTML = `<div class="p-3 text-xs text-rose-400">Failed to fetch preview.</div>`;
      }
    }

    function closeTablePreviewModal() {
      document.getElementById('tablePreviewModal').classList.add('hidden');
    }

    async function runAiQuery(execute = true) {
      const qInput = document.getElementById('aiQuestion');
      const question = qInput.value.trim();
      if (!question) {
        qInput.focus();
        return;
      }

      setLoading(true);
      resetAlerts();

      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: question,
            sql_url: state.sqlUrl,
            sql_user: state.sqlUser,
            sql_password: state.sqlPassword,
            provider: state.provider,
            api_key: state.apiKey,
            model: state.model,
            base_url: state.baseUrl,
            allow_writes: document.getElementById('allowWritesCheck').checked,
            allow_general: document.getElementById('allowGeneralCheck').checked,
            execute: execute,
          })
        });
        const data = await res.json();
        handleQueryResult(data);
      } catch (err) {
        showAlert('error', 'Execution Error', err.message);
      } finally {
        setLoading(false);
      }
    }

    async function runDirectSql() {
      const input = document.getElementById('rawSqlInput');
      const sql = input.value.trim();
      if (!sql) {
        input.focus();
        return;
      }

      setLoading(true);
      resetAlerts();

      try {
        const res = await fetch('/api/execute-sql', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sql: sql,
            sql_url: state.sqlUrl,
            sql_user: state.sqlUser,
            sql_password: state.sqlPassword,
            allow_writes: document.getElementById('allowWritesCheck').checked,
          })
        });
        const data = await res.json();
        handleQueryResult(data);
      } catch (err) {
        showAlert('error', 'Execution Error', err.message);
      } finally {
        setLoading(false);
      }
    }

    function handleQueryResult(data) {
      state.currentResults = data;
      state.currentSql = data.sql || null;

      const sqlView = document.getElementById('sqlViewContainer');
      const sqlCode = document.getElementById('sqlCodeBlock');
      const copyBtn = document.getElementById('btnCopySql');
      const exportBtn = document.getElementById('btnExportCsv');
      const retriesTag = document.getElementById('retriesUsedTag');
      const demoBanner = document.getElementById('demoNoticeBanner');
      const demoText = document.getElementById('demoNoticeText');

      if (data.is_demo_fallback) {
        demoBanner.classList.remove('hidden');
        if (data.message) demoText.textContent = data.message;
      } else {
        demoBanner.classList.add('hidden');
      }

      // General-knowledge answer (question not answerable from the DB)
      const generalView = document.getElementById('generalAnswerContainer');
      const generalText = document.getElementById('generalAnswerText');
      const resultBadgeEarly = document.getElementById('resultBadge');
      if (data.general_answer) {
        generalView.classList.remove('hidden');
        generalText.innerHTML = renderMarkdown(data.general_answer);
        resultBadgeEarly.classList.remove('hidden');
        resultBadgeEarly.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20';
        resultBadgeEarly.textContent = 'Answered';
        showEmptyTable("Answered from the model's general knowledge (not from the database).");
        return;
      }
      generalView.classList.add('hidden');

      if (data.sql) {
        sqlView.classList.remove('hidden');
        sqlCode.textContent = data.sql;
        delete sqlCode.dataset.highlighted;
        hljs.highlightElement(sqlCode);
        copyBtn.classList.remove('hidden');

        if (data.retries_used > 0) {
          retriesTag.textContent = `Auto self-healed in ${data.retries_used} retry loop(s)`;
        } else {
          retriesTag.textContent = '';
        }
      } else {
        sqlView.classList.add('hidden');
        copyBtn.classList.add('hidden');
      }

      // Badges
      const resultBadge = document.getElementById('resultBadge');
      const timeBadge = document.getElementById('executionTimeBadge');
      const pmBadge = document.getElementById('providerModelBadge');

      resultBadge.classList.remove('hidden');
      timeBadge.classList.remove('hidden');
      if (data.execution_time) {
        timeBadge.textContent = `${data.execution_time}s`;
      }
      if (data.provider && data.model) {
        pmBadge.textContent = `${data.provider} / ${data.model}`;
        pmBadge.classList.remove('hidden');
      } else {
        pmBadge.classList.add('hidden');
      }

      if (data.success) {
        resultBadge.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
        resultBadge.textContent = `${data.row_count || 0} rows`;

        if (data.columns && data.columns.length > 0) {
          exportBtn.classList.remove('hidden');
          renderMainTable(data.columns, data.rows);
        } else {
          exportBtn.classList.add('hidden');
          showEmptyTable("Statement executed successfully with no returned rows.");
        }
      } else {
        exportBtn.classList.add('hidden');
        resultBadge.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20';
        resultBadge.textContent = 'Failed';

        if (data.clarification) {
          showAlert('ambiguity', 'Clarification Required', data.clarification);
        } else {
          showAlert('error', data.error_type || 'Error', data.error || 'Failed to process query');
        }
        showEmptyTable("No results to display.");
      }
    }

    function renderMainTable(columns, rows) {
      const emptyState = document.getElementById('tableEmptyState');
      const tableWrapper = document.getElementById('tableWrapper');
      const thead = document.getElementById('dataThead');
      const tbody = document.getElementById('dataTbody');

      emptyState.classList.add('hidden');
      tableWrapper.classList.remove('hidden');

      thead.innerHTML = `<tr>${columns.map(c => `<th class="px-4 py-3 text-xs">${c}</th>`).join('')}</tr>`;
      tbody.innerHTML = rows.map((r, i) => `
        <tr class="${i % 2 === 0 ? 'bg-slate-900/40' : 'bg-slate-950/60'} hover:bg-slate-800/50 transition">
          ${columns.map(c => `<td class="px-4 py-2.5 text-xs text-slate-200 whitespace-nowrap">${r[c] !== null && r[c] !== undefined ? r[c] : '<span class="text-slate-600 italic">null</span>'}</td>`).join('')}
        </tr>
      `).join('');
    }

    function renderTableHtml(columns, rows) {
      return `
        <div class="overflow-x-auto rounded border border-slate-800">
          <table class="w-full text-left text-xs border-collapse">
            <thead class="bg-slate-800 text-slate-300 font-semibold border-b border-slate-700">
              <tr>${columns.map(c => `<th class="px-3 py-2">${c}</th>`).join('')}</tr>
            </thead>
            <tbody class="divide-y divide-slate-800 font-mono">
              ${rows.map((r, i) => `
                <tr class="${i % 2 === 0 ? 'bg-slate-900/60' : 'bg-slate-950'}">
                  ${columns.map(c => `<td class="px-3 py-2 text-slate-300 whitespace-nowrap">${r[c] !== null && r[c] !== undefined ? r[c] : '<span class="text-slate-600">null</span>'}</td>`).join('')}
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    function showEmptyTable(message) {
      const emptyState = document.getElementById('tableEmptyState');
      const tableWrapper = document.getElementById('tableWrapper');
      tableWrapper.classList.add('hidden');
      emptyState.classList.remove('hidden');
      emptyState.innerHTML = `<p class="text-xs text-slate-500 py-6">${message}</p>`;
    }

    function renderMarkdown(text) {
      const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      const inline = (s) => esc(s)
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/(^|[^*])\*([^*\s][^*]*)\*(?![*\w])/g, '$1<em>$2</em>')
        .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-slate-800 text-indigo-300 font-mono">$1</code>');

      const lines = String(text || '').split(/\\r?\\n/);;
      let html = '';
      let listOpen = false;
      const closeList = () => { if (listOpen) { html += '</ul>'; listOpen = false; } };

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        // Markdown table: current line has pipes and next is a separator row
        if (/^\|.*\|$/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && lines[i + 1].includes('-')) {
          closeList();
          const parseRow = (r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
          const headers = parseRow(line);
          i++; // skip separator
          const rows = [];
          while (i + 1 < lines.length && /^\s*\|.*\|\s*$/.test(lines[i + 1])) {
            i++;
            rows.push(lines[i]);
          }
          html += '<div class="overflow-x-auto my-3 rounded-lg border border-slate-800">' +
            '<table class="min-w-full text-xs"><thead class="bg-slate-800/80"><tr>' +
            headers.map((h) => `<th class="px-3 py-2 text-left font-semibold text-slate-300 whitespace-nowrap">${inline(h)}</th>`).join('') +
            '</tr></thead><tbody>' +
            rows.map((r, idx) => '<tr class="' + (idx % 2 ? 'bg-slate-900/40 ' : 'bg-slate-900/60 ') + 'border-t border-slate-800">' +
              parseRow(r).map((c) => `<td class="px-3 py-2 text-slate-200">${inline(c)}</td>`).join('') + '</tr>').join('') +
            '</tbody></table></div>';
          continue;
        }
        if (!line) { closeList(); continue; }
        const h = line.match(/^(#{1,6})\s+(.*)$/);
        if (h) {
          closeList();
          const level = Math.min(h[1].length + 2, 5);
          html += `<h${level} class="font-semibold text-slate-100 mt-2 mb-1">${inline(h[2])}</h${level}>`;
          continue;
        }
        const bullet = line.match(/^(?:[-*•])\s+(.*)$/);
        if (bullet) {
          if (!listOpen) { html += '<ul class="list-disc list-inside space-y-1 my-2 text-slate-200">'; listOpen = true; }
          html += `<li>${inline(bullet[1])}</li>`;
          continue;
        }
        closeList();
        html += `<p class="my-1.5">${inline(line)}</p>`;
      }
      closeList();
      return html;
    }

    function showAlert(type, title, message) {
      const banner = document.getElementById('alertBanner');
      banner.classList.remove('hidden');

      if (type === 'ambiguity') {
        banner.className = 'p-4 border-b border-amber-900/50 bg-amber-950/30 text-amber-300 text-xs flex items-start gap-3';
        banner.innerHTML = `
          <i data-lucide="help-circle" class="w-5 h-5 shrink-0 text-amber-400 mt-0.5"></i>
          <div>
            <div class="font-bold text-amber-200 mb-0.5">${title}</div>
            <div class="leading-relaxed">${message}</div>
          </div>
        `;
      } else {
        banner.className = 'p-4 border-b border-rose-900/50 bg-rose-950/30 text-rose-300 text-xs flex items-start gap-3';
        banner.innerHTML = `
          <i data-lucide="alert-triangle" class="w-5 h-5 shrink-0 text-rose-400 mt-0.5"></i>
          <div>
            <div class="font-bold text-rose-200 mb-0.5">${title}</div>
            <div class="leading-relaxed font-mono break-all">${message}</div>
          </div>
        `;
      }
      lucide.createIcons();
    }

    function resetAlerts() {
      document.getElementById('alertBanner').classList.add('hidden');
      document.getElementById('demoNoticeBanner').classList.add('hidden');
    }

    function setLoading(isLoading) {
      const btn = document.getElementById('btnAskAi');
      if (isLoading) {
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin"></i> Processing...`;
      } else {
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="sparkles" class="w-3.5 h-3.5"></i> Ask & Execute`;
      }
      lucide.createIcons();
    }

    function copySql() {
      if (!state.currentSql) return;
      navigator.clipboard.writeText(state.currentSql);
      const btn = document.getElementById('btnCopySql');
      btn.innerHTML = `<i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400"></i> Copied!`;
      lucide.createIcons();
      setTimeout(() => {
        btn.innerHTML = `<i data-lucide="copy" class="w-3.5 h-3.5"></i> Copy SQL`;
        lucide.createIcons();
      }, 2000);
    }

    function exportCsv() {
      if (!state.currentResults || !state.currentResults.rows) return;
      const { columns, rows } = state.currentResults;
      let csv = columns.join(',') + '\\n';
      rows.forEach(r => {
        csv += columns.map(c => {
          let val = r[c] !== null && r[c] !== undefined ? String(r[c]) : '';
          if (val.includes(',') || val.includes('"') || val.includes('\\n')) {
            val = '"' + val.replace(/"/g, '""') + '"';
          }
          return val;
        }).join(',') + '\\n';
      });

      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `query_result_${Date.now()}.csv`;
      a.click();
    }

    function toggleConfigModal(show) {
      const modal = document.getElementById('configModal');
      if (show) {
        document.getElementById('cfgProvider').value = state.provider;
        document.getElementById('cfgSqlUrl').value = state.sqlUrl;
        document.getElementById('cfgSqlUser').value = state.sqlUser;
        document.getElementById('cfgSqlPassword').value = state.sqlPassword;
        document.getElementById('cfgModalError').classList.add('hidden');
        onProviderChanged();
        modal.classList.remove('hidden');
      } else {
        modal.classList.add('hidden');
      }
    }

    function applyPreset(preset) {
      if (preset === 'sqlite') {
        document.getElementById('cfgSqlUrl').value = 'sqlite:///./demo.db';
        document.getElementById('cfgSqlUser').value = 'demo';
        document.getElementById('cfgSqlPassword').value = '';
      } else if (preset === 'postgres') {
        document.getElementById('cfgSqlUrl').value = 'postgresql://localhost:5432/mydb';
        document.getElementById('cfgSqlUser').value = 'postgres';
        document.getElementById('cfgSqlPassword').value = '';
      } else if (preset === 'mysql') {
        document.getElementById('cfgSqlUrl').value = 'mysql://localhost:3306/mydb';
        document.getElementById('cfgSqlUser').value = 'root';
        document.getElementById('cfgSqlPassword').value = '';
      }
    }

    async function testAndConnect() {
      const errBox = document.getElementById('cfgModalError');
      errBox.classList.add('hidden');

      const url = document.getElementById('cfgSqlUrl').value;
      const user = document.getElementById('cfgSqlUser').value;
      const pwd = document.getElementById('cfgSqlPassword').value;

      try {
        const res = await fetch('/api/connect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sql_url: url, sql_user: user, sql_password: pwd })
        });
        const data = await res.json();
        if (data.success) {
          alert('Connection Successful! Found ' + data.tables.length + ' table(s).');
        } else {
          errBox.textContent = data.error || 'Connection failed.';
          errBox.classList.remove('hidden');
        }
      } catch (e) {
        errBox.textContent = 'Error: ' + e.message;
        errBox.classList.remove('hidden');
      }
    }

    function saveConfigModal() {
      const p = document.getElementById('cfgProvider').value;
      const m = document.getElementById('cfgModel').value.trim();
      const u = document.getElementById('cfgBaseUrl').value.trim();
      const k = document.getElementById('cfgApiKey').value.trim();

      state.provider = p;
      state.model = m;
      state.baseUrl = u;
      state.apiKey = k;
      state.sqlUrl = document.getElementById('cfgSqlUrl').value.trim();
      state.sqlUser = document.getElementById('cfgSqlUser').value.trim();
      state.sqlPassword = document.getElementById('cfgSqlPassword').value;

      localStorage.setItem('qn_provider', p);
      localStorage.setItem('qn_model_' + p, m);
      localStorage.setItem('qn_url_' + p, u);
      if (k) localStorage.setItem('qn_key_' + p, k);
      localStorage.setItem('qn_sql_url', state.sqlUrl);
      localStorage.setItem('qn_sql_user', state.sqlUser);
      localStorage.setItem('qn_sql_password', state.sqlPassword);

      // Clean up legacy contaminated keys
      localStorage.removeItem('qn_base_url');
      localStorage.removeItem('qn_model');

      updateHeaderBadge();
      toggleConfigModal(false);
      refreshSchema();
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting QueryNexus Web Studio at http://localhost:{port}")
    uvicorn.run("web_app:app", host="127.0.0.1", port=port, reload=False)
