"""Command line: ask a question in a persona, or list the allowed tools.

    uv run gspaenli ask "What is the minimum health insurance premium in Lausanne for an adult?"
    uv run gspaenli ask "Quelle est la prime minimale à Lausanne ?" --persona buenzli --trace
    uv run gspaenli tools
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import Settings
from .llm import LLMClient
from .mcp_client import SwissMCP
from .orchestrator import Orchestrator
from .personas import load_personas


def build_llm(settings: Settings) -> LLMClient:
    if not settings.llm_base_url or not settings.llm_api_key:
        sys.exit("Set LLM_BASE_URL and LLM_API_KEY in .env (see .env.example)")
    return LLMClient(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.model_id,
        json_schema_supported=settings.llm_json_schema,
    )


async def _ask(question: str, persona_id: str, show_trace: bool) -> None:
    settings = Settings.from_env()
    personas = load_personas(settings.personas_dir)
    if persona_id not in personas:
        sys.exit(f"Unknown persona {persona_id!r}. Choose from: {', '.join(personas)}")
    llm = build_llm(settings)
    if not await llm.is_ready():
        print("The model is waking up (cold start, about 2 to 3 minutes)...", file=sys.stderr)
    async with SwissMCP(settings.mcp_dir) as mcp:
        turn = await Orchestrator(llm, mcp, personas, settings.mcp_dir).answer(question, persona_id)
    print(turn.answer)
    t = turn.trace
    print(f"\n[tool: {t.tool or 'none'} | lang: {t.lang} | {t.latency_s.get('total')}s | {t.model_id}]")
    if t.sources:
        print("[sources: " + ", ".join(t.sources) + "]")
    if show_trace:
        print(json.dumps(t.to_dict(), indent=2, ensure_ascii=False))


async def _tools() -> None:
    settings = Settings.from_env()
    async with SwissMCP(settings.mcp_dir) as mcp:
        for name in sorted(mcp.tools):
            print(name)


def main() -> None:
    parser = argparse.ArgumentParser(prog="gspaenli", description="Swiss answers, your choice of personality.")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--persona", default="newcomer")
    ask.add_argument("--trace", action="store_true", help="print the full trace as JSON")
    sub.add_parser("tools", help="list the allow-listed MCP tools")
    args = parser.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if args.command == "ask":
        asyncio.run(_ask(args.question, args.persona, args.trace))
    else:
        asyncio.run(_tools())


if __name__ == "__main__":
    main()
