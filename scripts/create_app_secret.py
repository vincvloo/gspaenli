"""Create the Modal secret "gspaenli-app" from your local .env, without printing the key.

    uv run python scripts/create_app_secret.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gspaenli.config import Settings  # noqa: E402


def main() -> int:
    s = Settings.from_env()
    if not s.llm_base_url or not s.llm_api_key:
        print("LLM_BASE_URL and LLM_API_KEY must be set in .env")
        return 1
    values = {
        "LLM_BASE_URL": s.llm_base_url,
        "LLM_API_KEY": s.llm_api_key,
        "MODEL_ID": s.model_id,
        "LLM_JSON_SCHEMA": "true" if s.llm_json_schema else "false",
    }
    result = subprocess.run(
        ["modal", "secret", "create", "gspaenli-app", *(f"{k}={v}" for k, v in values.items()), "--force"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Modal secret update failed:", result.stderr.replace(s.llm_api_key, "***"))
        return 1
    print(f"Secret 'gspaenli-app' saved: {', '.join(values)} (model {s.model_id}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
