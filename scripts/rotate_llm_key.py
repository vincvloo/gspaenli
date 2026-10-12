"""Generate a new LLM_API_KEY, store it in .env and in the Modal secret "gspaenli-llm".

The key is never printed. Run it yourself:  uv run python scripts/rotate_llm_key.py
Then redeploy the LLM server so new containers use it.
"""

import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"


def main() -> int:
    if not ENV.exists():
        shutil.copy(ROOT / ".env.example", ENV)
    key = secrets.token_urlsafe(32)

    result = subprocess.run(
        ["modal", "secret", "create", "gspaenli-llm", f"LLM_API_KEY={key}", "--force"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Modal secret update failed:", result.stderr.replace(key, "***"))
        return 1

    text = ENV.read_text(encoding="utf-8")
    line = f"LLM_API_KEY={key}"
    if re.search(r"^LLM_API_KEY=.*$", text, flags=re.M):
        text = re.sub(r"^LLM_API_KEY=.*$", lambda _: line, text, flags=re.M)
    else:
        text += f"\n{line}\n"
    ENV.write_text(text, encoding="utf-8")

    print(f"New key ({len(key)} chars) saved to .env and Modal secret 'gspaenli-llm'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
