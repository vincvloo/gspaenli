"""One chat turn: route -> resolve arguments -> one MCP tool call -> answer in persona."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .llm import ChatModel
from .mcp_client import ToolInfo, ToolResult
from .personas import Persona
from .prompts import build_answer_messages
from .resolver import (
    PREMIUM_TOOL,
    drop_named_optional,
    repair_choice,
    reroute_invented_coordinates,
    resolve,
    result_problem,
    router_tools,
    summarize_premiums,
)
from .router import route

ANSWER_MAX_TOKENS = 500
ANSWER_TEMPERATURE = 0.7
# Small models sometimes cite the prompt's own label, e.g. "Source: ... (TOOL RESULT)".
PROMPT_LEAK = re.compile(r"\s*\(?\bTOOL RESULT\b\)?", flags=re.I)


class MCPLike(Protocol):
    tools: dict[str, ToolInfo]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


@dataclass
class Trace:
    persona: str
    model_id: str
    lang: str = "en"
    tool: str | None = None
    routed_arguments: dict[str, Any] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra_calls: list[str] = field(default_factory=list)
    route_attempts: int = 0
    fallback: bool = False
    tool_error: str | None = None
    truncated: bool = False
    latency_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    answer: str
    trace: Trace
    tool_text: str = ""  # what the answer step saw; kept for evals


class Orchestrator:
    def __init__(self, llm: ChatModel, mcp: MCPLike, personas: dict[str, Persona], mcp_dir: Path) -> None:
        self.llm = llm
        self.mcp = mcp
        self.personas = personas
        self.mcp_dir = Path(mcp_dir)

    async def answer(
        self, question: str, persona_id: str, history: list[dict[str, str]] | None = None
    ) -> Turn:
        persona = self.personas[persona_id]
        trace = Trace(persona=persona_id, model_id=self.llm.model_id)
        started = time.perf_counter()

        t = time.perf_counter()
        chosen = await route(self.llm, question, persona, router_tools(self.mcp.tools), history)
        trace.latency_s["route"] = round(time.perf_counter() - t, 2)
        trace.lang, trace.tool, trace.routed_arguments = chosen.lang, chosen.tool, chosen.arguments
        trace.route_attempts, trace.fallback = chosen.attempts, chosen.fallback

        tool_text = ""
        if chosen.tool and (rerouted := reroute_invented_coordinates(chosen.tool, chosen.arguments, question)):
            chosen.tool, chosen.arguments, note = rerouted
            trace.tool = chosen.tool
            trace.notes.append(note)
        if chosen.tool:
            t = time.perf_counter()
            tool_text = await self._call_tool(chosen.tool, chosen.arguments, trace, question)
            trace.latency_s["tool"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        messages = build_answer_messages(
            question, persona, trace.lang, trace.tool, trace.arguments, tool_text, trace.notes, history
        )
        text = await self.llm.chat(messages, max_tokens=ANSWER_MAX_TOKENS, temperature=ANSWER_TEMPERATURE)
        text = PROMPT_LEAK.sub("", text).strip()
        trace.latency_s["answer"] = round(time.perf_counter() - t, 2)
        trace.latency_s["total"] = round(time.perf_counter() - started, 2)
        return Turn(answer=text, trace=trace, tool_text=tool_text)

    async def _call_tool(self, name: str, arguments: dict[str, Any], trace: Trace, question: str) -> str:
        resolution = await resolve(name, arguments, self.mcp.tools, self.mcp, self.mcp_dir, question)
        trace.arguments = resolution.arguments
        trace.notes.extend(resolution.notes)
        trace.extra_calls.extend(resolution.extra_calls)
        if resolution.error:
            trace.tool_error = resolution.error
            return f"ERROR: {resolution.error}"
        try:
            result = await self.mcp.call(name, resolution.arguments)
            problem = result_problem(result.text) or (result.text if result.is_error else None)
            required = set(self.mcp.tools[name].input_schema.get("required", [])) if name in self.mcp.tools else set()
            repaired = problem and (
                repair_choice(resolution.arguments, problem, required)
                or drop_named_optional(resolution.arguments, problem, required)
            )
            if repaired:
                trace.arguments, note = repaired
                trace.notes.append(note)
                result = await self.mcp.call(name, trace.arguments)
                problem = result_problem(result.text) or (result.text if result.is_error else None)
        except Exception as err:  # the answer step explains missing data in character
            trace.tool_error = f"{type(err).__name__}: {err}"
            return f"ERROR: the tool call failed ({trace.tool_error})"
        trace.sources, trace.truncated = result.sources, result.truncated
        if problem:
            trace.tool_error = problem[:300]
            return f"ERROR: the lookup returned no answer ({trace.tool_error}). Do not answer from memory."
        return summarize_premiums(result.text) if name == PREMIUM_TOOL else result.text
