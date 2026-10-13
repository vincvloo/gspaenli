"""Step 1 of a turn: the model picks one tool and its arguments as strict JSON.

Works without native tool calling. JSON schema output is requested when the endpoint supports it;
the parser still tolerates code fences and surrounding prose. On failure: one retry, then
search_knowledge with the raw question.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import ChatModel
from .mcp_client import ToolInfo
from .personas import Persona
from .prompts import (
    SUPPORTED_LANGS,
    Lookup,
    build_next_step_messages,
    build_route_messages,
    detect_language,
    route_retry_message,
)

FALLBACK_TOOL = "search_knowledge"
ROUTE_MAX_TOKENS = 200


class RouteError(ValueError):
    pass


@dataclass
class Route:
    tool: str | None  # None means no tool
    arguments: dict[str, Any]
    lang: str
    attempts: int = 1
    fallback: bool = False
    raw: list[str] = field(default_factory=list)


def route_schema(tool_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": [*tool_names, "none"]},
            "arguments": {"type": "object"},
            "lang": {"type": "string", "enum": list(SUPPORTED_LANGS)},
        },
        "required": ["tool", "arguments", "lang"],
    }


def _first_json_object(text: str) -> Any:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise RouteError("no JSON object found")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as err:
        raise RouteError(f"invalid JSON ({err.msg})") from err
    return obj


def parse_route(text: str, allowed: set[str], question: str) -> Route:
    obj = _first_json_object(text)
    if not isinstance(obj, dict):
        raise RouteError("JSON is not an object")
    tool = obj.get("tool")
    if not isinstance(tool, str):
        raise RouteError('"tool" must be a string')
    tool = tool.strip()
    if tool.lower() in {"none", "null", ""}:
        tool_name: str | None = None
    elif tool in allowed:
        tool_name = tool
    else:
        raise RouteError(f'unknown tool "{tool}"')
    arguments = obj.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise RouteError('"arguments" must be an object')
    lang = obj.get("lang")
    return Route(tool=tool_name, arguments=arguments, lang=detect_language(question, lang))


async def route(
    llm: ChatModel,
    question: str,
    persona: Persona,
    tools: dict[str, ToolInfo],
    history: list[dict[str, str]] | None = None,
) -> Route:
    messages = build_route_messages(question, persona, tools, history)
    schema = route_schema(sorted(tools))
    raw: list[str] = []
    for attempt in (1, 2):
        text = await llm.chat(messages, max_tokens=ROUTE_MAX_TOKENS, temperature=0.0, json_schema=schema)
        raw.append(text)
        try:
            result = parse_route(text, set(tools), question)
            result.attempts, result.raw = attempt, raw
            return result
        except RouteError as err:
            messages = [*messages, {"role": "assistant", "content": text}, route_retry_message(str(err))]
    return Route(
        tool=FALLBACK_TOOL if FALLBACK_TOOL in tools else None,
        arguments={"query": question},
        lang=detect_language(question, None),
        attempts=2,
        fallback=True,
        raw=raw,
    )


NEXT_STEP_MAX_TOKENS = 100  # a short JSON object; long replies cost seconds


def next_step_schema(tool_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            # First on purpose: naming the gap before deciding helps small models.
            "missing": {"type": "string"},
            "done": {"type": "boolean"},
            "tool": {"type": "string", "enum": [*tool_names, "none"]},
            "arguments": {"type": "object"},
        },
        "required": ["missing", "done"],
    }


def parse_next_step(text: str, allowed: set[str]) -> tuple[str, dict[str, Any]] | None:
    """(tool, arguments) for one more lookup, or None to stop.

    Conservative on purpose: anything unclear means "done", so a confused model cannot loop.
    """
    try:
        obj = _first_json_object(text)
    except RouteError:
        return None
    if not isinstance(obj, dict) or obj.get("done") is not False:
        return None
    tool = obj.get("tool")
    arguments = obj.get("arguments") or {}
    if not isinstance(tool, str) or tool not in allowed or not isinstance(arguments, dict):
        return None
    return tool, arguments


NOTHING_MISSING = {"", "nothing", "none", "n/a", "-", "rien", "nichts", "niente", "nulla"}


def parse_missing(text: str) -> str | None:
    """What the model said is still missing, when it said it is not done."""
    try:
        obj = _first_json_object(text)
    except RouteError:
        return None
    if not isinstance(obj, dict) or obj.get("done") is not False:
        return None
    missing = str(obj.get("missing") or "").strip()
    return None if missing.lower().strip(".") in NOTHING_MISSING else missing


async def next_step(
    llm: ChatModel,
    question: str,
    tools: dict[str, ToolInfo],
    done: list[Lookup],
    uncovered: list[str] | None = None,
) -> tuple[tuple[str, dict[str, Any]] | None, str]:
    """Ask whether another lookup is needed. Returns (next lookup or None, raw reply)."""
    messages = build_next_step_messages(question, tools, done, uncovered)
    schema = next_step_schema(sorted(tools))
    text = await llm.chat(messages, max_tokens=NEXT_STEP_MAX_TOKENS, temperature=0.0, json_schema=schema)
    return parse_next_step(text, set(tools)), text
