"""One chat turn: route -> one or more MCP lookups -> answer in persona.

The number of lookups is dynamic: after each one the model says whether the question is
answered or names one more lookup, up to MAX_LOOKUPS. The MCP only executes calls; all
decisions and guards live here.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .links import strip_unsourced_links
from .llm import ChatModel
from .mcp_client import ToolInfo, ToolResult
from .personas import Persona
from .prompts import Lookup, build_answer_messages, build_no_data_messages
from .resolver import (
    HELPER_TOOLS,
    PREMIUM_TOOL,
    SEARCH_TOOLS,
    Resolution,
    drop_named_optional,
    expand_for_place,
    gap_is_covered,
    may_have_several_parts,
    other_named_choices,
    place_tool_for,
    repair_choice,
    reroute_invented_coordinates,
    resolve,
    result_problem,
    router_tools,
    search_is_relevant,
    summarize_premiums,
    topic_words,
    uncovered_places,
)
from .router import next_step, parse_missing, route

MAX_LOOKUPS = 3  # upper bound per question; most questions stop after one
ANSWER_MAX_TOKENS = 500
NO_DATA_MAX_TOKENS = 150  # short on purpose: less room to improvise
# A message with this many topic words is a question, not small talk.
QUESTION_TOPIC_WORDS = 3
ANSWER_TEMPERATURE = 0.7
# Small models sometimes cite the prompt's own label, e.g. "Source: ... (TOOL RESULT)".
PROMPT_LEAK = re.compile(r"\s*\(?\bTOOL RESULT( \d+ of \d+)?\b\)?", flags=re.I)


class MCPLike(Protocol):
    tools: dict[str, ToolInfo]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


@dataclass
class Step:
    """One MCP lookup, with what the guards did to it."""

    tool: str
    routed_arguments: dict[str, Any]
    arguments: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra_calls: list[str] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False
    latency_s: float = 0.0
    text: str = field(default="", repr=False)
    # Further options the model's value named, each still to be looked up (same tool).
    pending: list[dict[str, Any]] = field(default_factory=list, repr=False)
    for_missing: str | None = None  # the gap the model named, when this lookup was routed for it

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text")
        data.pop("pending")
        return data


@dataclass
class Trace:
    persona: str
    model_id: str
    lang: str = "en"
    # First lookup, kept at the top level for the UI and evals; all lookups are in `steps`.
    tool: str | None = None
    routed_arguments: dict[str, Any] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra_calls: list[str] = field(default_factory=list)
    route_attempts: int = 0
    fallback: bool = False
    tool_error: str | None = None  # set only when no lookup returned usable data
    truncated: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    missing_parts: list[str] = field(default_factory=list)  # gaps the model named
    removed_links: list[str] = field(default_factory=list)  # links in the answer not found in the data
    stop_reason: str = ""
    latency_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    answer: str
    trace: Trace
    tool_text: str = ""  # what the answer step saw; kept for evals


class Orchestrator:
    def __init__(
        self,
        llm: ChatModel,
        mcp: MCPLike,
        personas: dict[str, Persona],
        mcp_dir: Path,
        max_lookups: int = MAX_LOOKUPS,
    ) -> None:
        self.llm = llm
        self.mcp = mcp
        self.personas = personas
        self.mcp_dir = Path(mcp_dir)
        self.max_lookups = max(1, max_lookups)

    async def answer(
        self, question: str, persona_id: str, history: list[dict[str, str]] | None = None
    ) -> Turn:
        persona = self.personas[persona_id]
        trace = Trace(persona=persona_id, model_id=self.llm.model_id)
        started = time.perf_counter()
        tools = router_tools(self.mcp.tools)

        t = time.perf_counter()
        chosen = await route(self.llm, question, persona, tools, history)
        trace.latency_s["route"] = round(time.perf_counter() - t, 2)
        trace.lang, trace.routed_arguments = chosen.lang, chosen.arguments
        trace.route_attempts, trace.fallback = chosen.attempts, chosen.fallback

        tool, arguments = chosen.tool, chosen.arguments
        if tool is None and len(topic_words(question, self.mcp_dir)) >= QUESTION_TOPIC_WORDS:
            # Qwen answered a real question with no tool and improvised a URL. Search instead, so
            # the relevance check decides between a grounded answer and an honest "not found".
            tool, arguments = "search_knowledge", {"query": question}
            trace.notes.append("The router chose no tool for a question; searched saved pages instead")

        steps: list[Step] = []
        for_missing: str | None = None
        trace.stop_reason = "no lookup needed" if tool is None else ""
        while tool is not None:
            if rerouted := reroute_invented_coordinates(tool, arguments, question):
                tool, arguments, note = rerouted
                trace.notes.append(note)
            if any(s.tool == tool and s.routed_arguments == arguments for s in steps):
                trace.stop_reason = "the model asked for a lookup it had already made"
                break
            step = await self._lookup(tool, arguments, question, steps)
            if step is None:
                trace.stop_reason = "the model asked for a lookup it had already made"
                break
            step.for_missing, for_missing = for_missing, None
            steps.append(step)
            if len(steps) >= self.max_lookups:
                trace.stop_reason = f"reached the limit of {self.max_lookups} lookups"
                break
            following = await self._choose_next(question, persona, tools, steps, trace)
            if following is None:
                trace.stop_reason = trace.stop_reason or "the model had enough"
                break
            tool, arguments, for_missing = following
        good_steps = [s for s in steps if not s.error]
        still_uncovered = uncovered_places(question, self._coverage(good_steps), self.mcp_dir) if steps else []
        if still_uncovered:
            # Without this, Apertus invented Geneva premiums it had never looked up.
            trace.notes.append(
                f"No lookup was made for {', '.join(still_uncovered)}: say it could not be checked, "
                "give no figures or facts for it"
            )
        for gap in trace.missing_parts:
            if not any(s.for_missing == gap and not s.error for s in steps):
                # Apertus filled such gaps from memory ("holidays usually start late September").
                trace.notes.append(f"Not looked up: {gap}. Say it could not be checked, give no facts for it")
        self._fill_trace(trace, steps)

        good = [s for s in steps if not s.error]
        lookups: list[Lookup] = [(s.tool, s.arguments, s.text) for s in good]
        # Where a place is does not answer a question about something else (Qwen invented waste
        # rules from a location lookup for Bern).
        only_helpers = bool(good) and all(s.tool in HELPER_TOOLS for s in good)
        no_data = bool(steps) and (not good or (only_helpers and bool(topic_words(question, self.mcp_dir))))
        t = time.perf_counter()
        if no_data:
            messages = build_no_data_messages(question, persona, trace.lang)
            max_tokens = NO_DATA_MAX_TOKENS
        else:
            messages = build_answer_messages(question, persona, trace.lang, lookups, trace.notes, history)
            max_tokens = ANSWER_MAX_TOKENS
        text = await self.llm.chat(messages, max_tokens=max_tokens, temperature=ANSWER_TEMPERATURE)
        text = PROMPT_LEAK.sub("", text).strip()
        data = "\n".join([*(s.text for s in good), *trace.sources]) if not no_data else ""
        text, trace.removed_links = strip_unsourced_links(text, data)
        trace.latency_s["answer"] = round(time.perf_counter() - t, 2)
        trace.latency_s["total"] = round(time.perf_counter() - started, 2)
        if len(good) > 1:
            tool_text = "\n\n".join(f"[{s.tool}] {s.text}" for s in good)
        else:
            tool_text = good[0].text if good else ""
        return Turn(answer=text, trace=trace, tool_text=tool_text)

    async def _choose_next(
        self, question: str, persona: Persona, tools: dict[str, ToolInfo], steps: list[Step], trace: Trace
    ) -> tuple[str, dict[str, Any], str | None] | None:
        """The next lookup (tool, arguments, gap it fills), or None when the question is covered:

        1. topics the last lookup's value named but could not cover (referendum and initiative),
        2. the model's own choice,
        3. the same place-based lookup for a place in the question not covered yet,
        4. the router on whatever is still missing (the model's words or an uncovered place).
        Steps 3 and 4 exist because Apertus often noticed the gap but named no tool.
        """
        last = steps[-1]
        if last.pending:
            arguments = last.pending.pop(0)
            trace.notes.append(f"Also looked up {json.dumps(arguments, ensure_ascii=False)}, named in the question")
            return last.tool, arguments, None
        uncovered = uncovered_places(question, self._coverage(steps), self.mcp_dir)
        if not uncovered and not may_have_several_parts(question, self.mcp_dir):
            trace.stop_reason = "single-part question"  # skip the extra model call
            return None

        # Arguments as the model wrote them: shown resolved codes, Apertus copied the format
        # and invented municipality_code 6600 for Geneva (it is 6621).
        done: list[Lookup] = [(s.tool, s.routed_arguments, s.text or f"(no answer: {s.error})") for s in steps]
        started = time.perf_counter()
        following, raw = await next_step(self.llm, question, tools, done, uncovered)
        self._add_latency(trace, "next_step", started)
        if following is not None:
            return following[0], following[1], parse_missing(raw)

        if uncovered and (expanded := expand_for_place(last.tool, last.routed_arguments, uncovered[0])):
            trace.notes.append(f"Added a lookup for {uncovered[0]}, named in the question")
            return expanded[0], expanded[1], None
        if uncovered and (place_tool := place_tool_for(question, {s.tool for s in steps})):
            if expanded := expand_for_place(place_tool, {}, uncovered[0]):
                trace.notes.append(f"Added a lookup for {uncovered[0]}, named in the question")
                return expanded[0], expanded[1], None

        missing = parse_missing(raw)
        if missing and _restates(missing, question):
            missing = None  # Apertus sometimes repeats the whole question as "missing"
        if missing and gap_is_covered(missing, self._coverage([s for s in steps if not s.error]), self.mcp_dir):
            missing = None  # already answered by a lookup; chasing it wastes a lookup
        missing = missing or (", ".join(uncovered) if uncovered else None)
        if missing:
            trace.missing_parts.append(missing)
            # Only tools not used yet: with all tools on offer, Apertus routed the gap back to the
            # tool it had already used. Same tool for another place is expand_for_place's job.
            used = {s.tool for s in steps}
            fresh = {name: info for name, info in tools.items() if name not in used}
            if not fresh:
                return None
            # A focused sub-question routes better than the full question plus a hint: with the
            # full question Apertus picked housing and weather for "vacanze autunnali in Ticino".
            sub_question = missing if len(missing.split()) >= 3 else f"{question}\nStill missing: {missing}"
            started = time.perf_counter()
            rerouted = await route(self.llm, sub_question, persona, fresh)
            self._add_latency(trace, "next_step", started)
            if rerouted.tool and not rerouted.fallback:
                trace.notes.append(f"Looked up what was still missing: {missing}")
                return rerouted.tool, rerouted.arguments, missing
        return None

    @staticmethod
    def _add_latency(trace: Trace, key: str, started: float) -> None:
        trace.latency_s[key] = round(trace.latency_s.get(key, 0.0) + time.perf_counter() - started, 2)

    @staticmethod
    def _coverage(steps: list[Step]) -> list[Lookup]:
        """Both the model's arguments ("Genf") and the resolved ones (code 6621) count as covered."""
        return [(s.tool, {**s.routed_arguments, **s.arguments}, s.text) for s in steps]

    @staticmethod
    def _fill_trace(trace: Trace, steps: list[Step]) -> None:
        """Top-level trace fields summarise all steps; failed steps stay visible in `steps`."""
        trace.steps = [s.summary() for s in steps]
        if not steps:
            return
        first = steps[0]
        trace.tool, trace.arguments = first.tool, first.arguments
        for step in steps:
            trace.notes.extend(n for n in step.notes if n not in trace.notes)
            trace.extra_calls.extend(step.extra_calls)
            trace.sources.extend(u for u in step.sources if u not in trace.sources)
            trace.truncated = trace.truncated or step.truncated
        if all(s.error for s in steps):
            trace.tool_error = first.error if len(steps) == 1 else "; ".join(s.error or "" for s in steps)
        trace.latency_s["tool"] = round(sum(s.latency_s for s in steps), 2)

    async def _lookup(
        self, name: str, arguments: dict[str, Any], question: str, previous: list[Step]
    ) -> Step | None:
        """One lookup, or None when it resolves to a lookup already made (no MCP call)."""
        step = Step(tool=name, routed_arguments=arguments)
        started = time.perf_counter()
        try:
            resolution: Resolution = await resolve(
                name, arguments, self.mcp.tools, self.mcp, self.mcp_dir, question
            )
            if any(p.tool == name and p.arguments == resolution.arguments for p in previous):
                # Apertus repeated lookups with arguments that resolve to the same call.
                return None
            step.arguments = resolution.arguments
            step.notes.extend(resolution.notes)
            step.extra_calls.extend(resolution.extra_calls)
            if resolution.error:
                step.error = resolution.error
            else:
                await self._call_tool(step, question)
        finally:
            step.latency_s = round(time.perf_counter() - started, 2)
        return step

    async def _call_tool(self, step: Step, question: str) -> None:
        name = step.tool
        try:
            result = await self.mcp.call(name, step.arguments)
            problem = result_problem(result.text) or (result.text if result.is_error else None)
            required = set(self.mcp.tools[name].input_schema.get("required", [])) if name in self.mcp.tools else set()
            repaired = problem and (
                repair_choice(step.arguments, problem, required)
                or drop_named_optional(step.arguments, problem, required)
            )
            if repaired:
                step.pending = other_named_choices(step.arguments, problem)
                step.arguments, note = repaired
                step.notes.append(note)
                result = await self.mcp.call(name, step.arguments)
                problem = result_problem(result.text) or (result.text if result.is_error else None)
        except Exception as err:  # the answer step explains missing data in character
            step.error = f"{type(err).__name__}: {err}"
            return
        step.sources, step.truncated = result.sources, result.truncated
        if not problem and name in SEARCH_TOOLS and not search_is_relevant(
            question, step.arguments, result.text, self.mcp_dir
        ):
            problem = "No saved official page matches the topic of the question"
            step.sources = []  # the pages found are about something else
        if problem:
            step.error = problem[:300]
            return
        step.text = summarize_premiums(result.text) if name == PREMIUM_TOOL else result.text


def _restates(missing: str, question: str) -> bool:
    """True when the model's "missing" text is essentially the whole question."""
    import difflib

    a, b = missing.lower().strip(" ?.!"), question.lower().strip(" ?.!")
    return len(a) > 0.8 * len(b) and difflib.SequenceMatcher(None, a, b).ratio() > 0.8
