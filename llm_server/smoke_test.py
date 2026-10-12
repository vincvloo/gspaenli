"""Smoke test for the deployed LLM server. Reads LLM_BASE_URL, LLM_API_KEY, MODEL_ID from env/.env.

    uv run python llm_server/smoke_test.py
"""

import json
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, ".")
from gspaenli.config import Settings  # noqa: E402


def request(url: str, key: str | None, body: dict | None = None, timeout: float = 600) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, {}


def main() -> int:
    s = Settings.from_env()
    if not s.llm_base_url or not s.llm_api_key:
        print("Set LLM_BASE_URL and LLM_API_KEY in .env")
        return 1
    base = s.llm_base_url.rstrip("/")
    ok = True

    status, _ = request(f"{base}/models", key=None)
    print(f"no token -> {status} (expect 401)")
    ok &= status == 401

    started = time.perf_counter()
    status, body = request(f"{base}/models", key=s.llm_api_key)
    models = [m["id"] for m in body.get("data", [])]
    print(f"models -> {status} {models} ({time.perf_counter() - started:.1f}s, includes cold start)")
    ok &= s.model_id in models

    for question in [
        "Réponds en une phrase : quelle est la capitale de la Suisse ?",
        "Antworte in einem Satz: Was ist die Hauptstadt der Schweiz?",
    ]:
        started = time.perf_counter()
        status, body = request(
            f"{base}/chat/completions",
            key=s.llm_api_key,
            body={
                "model": s.model_id,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 120,
                "temperature": 0.2,
            },
        )
        elapsed = time.perf_counter() - started
        answer = body["choices"][0]["message"]["content"].strip() if status == 200 else ""
        tokens = body.get("usage", {}).get("completion_tokens", 0)
        print(f"chat -> {status} {tokens} tokens in {elapsed:.1f}s ({tokens / elapsed:.0f} tok/s): {answer}")
        ok &= status == 200 and bool(answer)

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
