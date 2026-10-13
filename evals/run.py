"""Run evals/questions.yaml through the real pipeline (route, resolve, MCP call, answer).

    uv run python evals/run.py                                  # model and endpoint from .env
    uv run python evals/run.py --model Qwen/Qwen3-8B --base-url https://...-qwen-serve.modal.run/v1

Writes evals/results/<model>.json and prints a Markdown summary. Uses the GPU: one run is
about 25 model calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gspaenli.config import Settings  # noqa: E402
from gspaenli.llm import LLMClient  # noqa: E402
from gspaenli.mcp_client import SwissMCP  # noqa: E402
from gspaenli.orchestrator import Orchestrator  # noqa: E402
from gspaenli.personas import load_personas  # noqa: E402
from gspaenli.prompts import guess_language  # noqa: E402

HERE = Path(__file__).resolve().parent


def answer_language(text: str) -> str:
    """Language of the answer itself, ignoring URLs and quoted official terms."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    return guess_language(text)


async def run(model_id: str, base_url: str, api_key: str, json_schema: bool) -> dict:
    settings = Settings.from_env()
    personas = load_personas(settings.personas_dir)
    questions = yaml.safe_load((HERE / "questions.yaml").read_text(encoding="utf-8"))
    llm = LLMClient(base_url, api_key, model_id, json_schema_supported=json_schema)

    print(f"Model {model_id}: warming up...", file=sys.stderr)
    started = time.perf_counter()
    await llm.chat([{"role": "user", "content": "Hi"}], max_tokens=5, temperature=0.0)
    warmup_s = round(time.perf_counter() - started, 1)

    rows = []
    async with SwissMCP(settings.mcp_dir) as mcp:
        orchestrator = Orchestrator(llm, mcp, personas, settings.mcp_dir)
        for q in questions:
            try:
                turn = await orchestrator.answer(q["question"], q["persona"])
                t, answer, error, tool_text = turn.trace, turn.answer, None, turn.tool_text
            except Exception as err:  # keep going, record the failure
                t, answer, error, tool_text = None, "", f"{type(err).__name__}: {err}", ""
            row = {
                "id": q["id"],
                "question": q["question"],
                "expected_tool": q["expected_tool"],
                "expected_lang": q["lang"],
                "tool": t.tool if t else None,
                "arguments": t.arguments if t else {},
                "routed_lang": t.lang if t else None,
                "answer_lang": answer_language(answer) if answer else None,
                "fallback": t.fallback if t else None,
                "route_attempts": t.route_attempts if t else None,
                "tool_error": t.tool_error if t else error,
                "latency_s": t.latency_s.get("total") if t else None,
                "notes": t.notes if t else [],
                "answer": answer,
                "tool_result": tool_text[:1500],
            }
            row["tool_ok"] = row["tool"] == row["expected_tool"]
            row["lang_ok"] = row["answer_lang"] == row["expected_lang"]
            rows.append(row)
            mark = lambda ok: "ok " if ok else "XX "  # noqa: E731
            print(
                f"{mark(row['tool_ok'])}{mark(row['lang_ok'])}{q['id']:<22} tool={row['tool']} "
                f"lang={row['answer_lang']} {row['latency_s']}s",
                file=sys.stderr,
            )

    latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    summary = {
        "model": model_id,
        "questions": len(rows),
        "tool_accuracy": sum(r["tool_ok"] for r in rows) / len(rows),
        "language_accuracy": sum(r["lang_ok"] for r in rows) / len(rows),
        "router_fallbacks": sum(bool(r["fallback"]) for r in rows),
        "tool_errors": sum(bool(r["tool_error"]) for r in rows),
        "median_latency_s": round(statistics.median(latencies), 1) if latencies else None,
        "warmup_s": warmup_s,
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    s = Settings.from_env()
    model_id = args.model or s.model_id
    result = asyncio.run(run(model_id, args.base_url or s.llm_base_url, s.llm_api_key, s.llm_json_schema))

    out = HERE / "results" / f"{model_id.split('/')[-1]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    sm = result["summary"]
    print(
        f"| {sm['model']} | {sm['tool_accuracy']:.0%} | {sm['language_accuracy']:.0%} | "
        f"{sm['router_fallbacks']} | {sm['tool_errors']} | {sm['median_latency_s']} s |"
    )
    print(f"Saved {out}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    main()
