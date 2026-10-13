"""Gradio UI: persona cards, chat, a "How I answered" panel under every answer, guided tour.

Local run:  uv run python -m gspaenli.ui
On Modal:   modal_app.py mounts create_app() as an ASGI app.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no telemetry from a public demo

import gradio as gr  # noqa: E402

from .config import Settings  # noqa: E402
from .llm import LLMClient  # noqa: E402
from .mcp_client import SwissMCP  # noqa: E402
from .orchestrator import Orchestrator, Trace  # noqa: E402
from .personas import Persona, load_personas  # noqa: E402
from .ratelimit import RateLimiter  # noqa: E402
from .ui_assets import (  # noqa: E402
    BASE_CSS,
    MCP_URL,
    REPO_URL,
    AVATAR_COLORS,
    avatar_png,
    header_html,
    persona_css,
    persona_label,
    placeholder_md,
    status_html,
    theme,
    tour_head,
)

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
HISTORY_MESSAGES = 8
# The GPU scales down after 5 idle minutes; call it ready a bit less than that.
READY_WINDOW_S = 270

COLD_START_NOTICE = (
    "The model is waking up (it sleeps when idle to save GPU costs). "
    "The first answer takes about 2 minutes, the next ones a few seconds."
)
ERROR_MESSAGE = "Sorry, something went wrong on my side. Please try again in a minute."


def format_trace(trace: Trace) -> str:
    """Markdown for the collapsible "How I answered" panel."""
    lines = []
    if trace.tool:
        lines.append(f"**Tool:** `{trace.tool}` (Swiss MCP)")
    else:
        lines.append("**Tool:** none, no lookup was needed")
    if trace.arguments:
        lines.append(f"**Arguments:** `{json.dumps(trace.arguments, ensure_ascii=False)}`")
    if trace.notes:
        lines.append("**Assumptions:** " + "; ".join(trace.notes))
    if trace.extra_calls:
        lines.append("**Extra lookups:** " + ", ".join(trace.extra_calls))
    if trace.fallback:
        lines.append("**Note:** the router could not pick a tool, so this used a full-text search.")
    if trace.tool_error:
        lines.append(f"**Tool error:** {trace.tool_error}")
    if trace.sources:
        lines.append("**Sources:**\n" + "\n".join(f"- [{url}]({url})" for url in trace.sources))
    elif trace.tool:
        lines.append("**Sources:** none returned")
    lat = trace.latency_s
    steps = [f"{name} {lat[name]:.1f} s" for name in ("route", "tool", "answer") if name in lat]
    if steps:
        lines.append(f"**Latency:** {', '.join(steps)}, total {lat.get('total', 0):.1f} s")
    lines.append(f"**Model:** `{trace.model_id}`, **reply language:** {trace.lang}")
    return "\n\n".join(lines)


class Runtime:
    """Shared per container: personas, LLM client, one MCP subprocess, rate limits, model state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.personas = load_personas(settings.personas_dir)
        self.llm = LLMClient(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.model_id,
            json_schema_supported=settings.llm_json_schema,
        )
        self.limiter = RateLimiter()
        self.avatars = self._write_avatars(self.personas)
        self.last_ok = 0.0
        self.waking = False
        self._orchestrator: Orchestrator | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _write_avatars(personas: dict[str, Persona]) -> dict[str, str]:
        folder = Path(tempfile.mkdtemp(prefix="gspaenli-avatars-"))
        paths = {}
        for i, persona in enumerate(personas.values()):
            path = folder / f"{persona.id}.png"
            avatar_png(persona.name[0].upper(), AVATAR_COLORS[i % len(AVATAR_COLORS)], str(path))
            paths[persona.id] = str(path)
        return paths

    def model_state(self) -> str:
        """From what the app already knows. Never polls the GPU: any request would wake it."""
        if self.waking:
            return "waking"
        return "ready" if time.time() - self.last_ok < READY_WINDOW_S else "asleep"

    async def orchestrator(self) -> Orchestrator:
        async with self._lock:
            if self._orchestrator is None:
                mcp = SwissMCP(self.settings.mcp_dir)
                await mcp.start_in_background()
                missing = {p.id: mcp.missing_tools(p.preferred_tools) for p in self.personas.values()}
                if any(missing.values()):
                    log.warning("preferred tools not available on this MCP: %s", missing)
                self._orchestrator = Orchestrator(self.llm, mcp, self.personas, self.settings.mcp_dir)
            return self._orchestrator


def build_demo(runtime: Runtime) -> gr.Blocks:
    personas = runtime.personas
    first = next(iter(personas.values()))

    def status() -> str:
        ready_until_ms = int((runtime.last_ok + READY_WINDOW_S) * 1000)
        return status_html(runtime.model_state(), ready_until_ms)

    async def respond(message: str, persona_id: str, chat: list, history: list, session: dict):
        message = (message or "").strip()[:MAX_QUESTION_CHARS]
        if not message:
            yield chat, history, session, "", status()
            return
        chat = [*chat, {"role": "user", "content": message}]
        if blocked := runtime.limiter.check(session):
            yield [*chat, {"role": "assistant", "content": blocked}], history, session, "", status()
            return
        yield chat, history, session, "", status()

        if runtime.model_state() != "ready" and not await runtime.llm.is_ready():
            runtime.waking = True
            notice = {
                "role": "assistant",
                "content": COLD_START_NOTICE,
                "metadata": {"title": "Waking up the model", "status": "pending"},
            }
            yield [*chat, notice], history, session, "", status()
        try:
            orchestrator = await runtime.orchestrator()
            turn = await orchestrator.answer(message, persona_id, history)
            runtime.last_ok = time.time()
        except Exception:
            log.exception("turn failed")
            runtime.waking = False
            yield [*chat, {"role": "assistant", "content": ERROR_MESSAGE}], history, session, "", status()
            return
        runtime.waking = False

        chat = [
            *chat,
            {"role": "assistant", "content": turn.answer},
            {
                "role": "assistant",
                "content": format_trace(turn.trace),
                "metadata": {"title": "How I answered", "status": "done"},
            },
        ]
        history = [*history, {"role": "user", "content": message}, {"role": "assistant", "content": turn.answer}]
        yield chat, history[-HISTORY_MESSAGES:], session, "", status()

    def switch_persona(persona_id: str):
        persona = personas[persona_id]
        chat = gr.update(
            value=[],
            placeholder=placeholder_md(persona),
            avatar_images=(None, runtime.avatars[persona.id]),
        )
        examples = [gr.update(value=q) for q in persona.example_questions]
        return chat, *examples, []

    with gr.Blocks(title="Gspänli", analytics_enabled=False) as demo:
        gr.HTML(header_html(), elem_id="gs-header")
        with gr.Row(elem_id="gs-toolbar"):
            status_pill = gr.HTML(status(), elem_id="gs-status")
            tour = gr.Button("Take the tour", size="sm", variant="secondary", elem_id="gs-tour-btn")
        persona = gr.Radio(
            choices=[(persona_label(p), p.id) for p in personas.values()],
            value=first.id,
            show_label=False,
            container=False,
            elem_id="gs-personas",
        )
        chatbot = gr.Chatbot(
            show_label=False,
            height=480,
            placeholder=placeholder_md(first),
            avatar_images=(None, runtime.avatars[first.id]),
            elem_id="gs-chat",
        )
        with gr.Row(elem_id="gs-examples"):
            example_buttons = [gr.Button(q, size="sm", variant="secondary") for q in first.example_questions]
        with gr.Row(elem_id="gs-input"):
            box = gr.Textbox(
                placeholder="Ask in English, French, German or Italian",
                show_label=False,
                container=False,
                scale=8,
                max_lines=4,
            )
            ask = gr.Button("Ask", variant="primary", scale=1, elem_id="gs-ask")
        new_chat = gr.Button("New conversation", size="sm", variant="secondary")
        gr.HTML(
            "<p class='gs-footnote'>Proof of concept. Answers can contain mistakes: check figures at the "
            f"linked source. Data: <a href='{MCP_URL}' target='_blank'>Swiss MCP</a>. "
            f"Model: Apertus 8B on Modal. <a href='{REPO_URL}' target='_blank'>Source code</a>.</p>"
        )

        history = gr.State([])
        session = gr.State({})
        outputs: list[Any] = [chatbot, history, session, box, status_pill]

        for trigger in (ask.click, box.submit):
            trigger(respond, [box, persona, chatbot, history, session], outputs, concurrency_limit=8)
        for button in example_buttons:
            button.click(respond, [button, persona, chatbot, history, session], outputs, concurrency_limit=8)
        persona.change(switch_persona, persona, [chatbot, *example_buttons, history])
        new_chat.click(lambda: ([], []), None, [chatbot, history])
        tour.click(None, js="() => window.gspaenliTour && window.gspaenliTour()")
        # Pill state on page load and after each turn. No server timer: an open tab would keep
        # the app container awake. The browser expires "ready" by itself (tour_head).
        demo.load(status, None, status_pill)
    return demo


def page_css(runtime: Runtime) -> str:
    return BASE_CSS + persona_css(list(runtime.personas.values()))


def create_app():
    """FastAPI app with the Gradio UI mounted at /, used by modal_app.py."""
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    logging.basicConfig(level=logging.INFO)
    runtime = Runtime(Settings.from_env())
    return mount_gradio_app(
        app=FastAPI(),
        blocks=build_demo(runtime),
        path="/",
        theme=theme(),
        css=page_css(runtime),
        head=tour_head(),
        footer_links=[],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runtime = Runtime(Settings.from_env())
    build_demo(runtime).launch(theme=theme(), css=page_css(runtime), head=tour_head())
