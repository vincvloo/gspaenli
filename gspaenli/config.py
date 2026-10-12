"""Runtime configuration from environment variables (and an optional local .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    model_id: str
    llm_json_schema: bool
    mcp_dir: Path
    personas_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        mcp_dir = Path(os.environ.get("MCP_DIR", "vendor/swiss-ai-week-mcp"))
        if not mcp_dir.is_absolute():
            mcp_dir = ROOT / mcp_dir
        return cls(
            llm_base_url=os.environ.get("LLM_BASE_URL", ""),
            llm_api_key=os.environ.get("LLM_API_KEY", ""),
            model_id=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
            # vLLM supports JSON schema output; set false for endpoints that do not.
            llm_json_schema=os.environ.get("LLM_JSON_SCHEMA", "true").lower() != "false",
            mcp_dir=mcp_dir,
            personas_dir=Path(os.environ.get("PERSONAS_DIR", ROOT / "personas")),
        )
