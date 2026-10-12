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
from .prompts import SUPPORTED_LANGS, build_route_messages, detect_language, route_retry_message

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
