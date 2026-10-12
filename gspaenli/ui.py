"""Gradio UI: persona picker, chat, and a "How I answered" panel under every answer.

Local run:  uv run python -m gspaenli.ui
On Modal:   modal_app.py mounts create_app() as an ASGI app.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no telemetry from a public demo

import gradio as gr  # noqa: E402

from .config import Settings
from .llm import LLMClient
from .mcp_client import SwissMCP
from .orchestrator import Orchestrator, Trace
from .personas import Persona, load_personas
from .ratelimit import RateLimiter

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
HISTORY_MESSAGES = 8

ONBOARDING = """# Gspänli
**Swiss answers, your choice of personality.**

Pick a personality and ask about life in Switzerland: health insurance, residence permits, housing, school holidays, political rights and more. Ask in English, French, German or Italian, and you get the answer in your language, even when the source is in another one.

Each answer is grounded in one lookup in official Swiss data through the [Swiss MCP server](https://github.com/dantevier/swiss-ai-week-mcp), and written by [Apertus](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509), the Swiss open model, running on [Modal](https://modal.com). Open **How I answered** under an answer to see the tool, its arguments and the source.

This is a proof of concept: check important figures at the linked source. The model sleeps when nobody uses it, so the first answer can take about 2 minutes."""

COLD_START_NOTICE = (
    "The model is waking up (it sleeps when idle to save GPU costs). "
    "The first answer takes about 2 minutes, the next ones a few seconds."
)
ERROR_MESSAGE = "Sorry, something went wrong on my side. Please try again in a minute."

CSS = """
.gradio-container { max-width: 900px !important; margin: auto; }
.persona-card p { margin: 0.2em 0; }
"""


def persona_card(persona: Persona) -> str:
    return f"**{persona.name}**: {persona.tagline}"


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
    """Shared per container: personas, LLM client, one MCP subprocess, rate limits."""

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
        self._orchestrator: Orchestrator | None = None
        self._lock = asyncio.Lock()

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

    async def respond(message: str, persona_id: str, chat: list, history: list, session: dict):
        message = (message or "").strip()[:MAX_QUESTION_CHARS]
        if not message:
            yield chat, history, session, ""
            return
        chat = [*chat, {"role": "user", "content": message}]
        if blocked := runtime.limiter.check(session):
            yield [*chat, {"role": "assistant", "content": blocked}], history, session, ""
            return
        yield chat, history, session, ""

        if not await runtime.llm.is_ready():
            notice = {"role": "assistant", "content": COLD_START_NOTICE, "metadata": {"title": "Waking up", "status": "pending"}}
            yield [*chat, notice], history, session, ""
        try:
            orchestrator = await runtime.orchestrator()
            turn = await orchestrator.answer(message, persona_id, history)
        except Exception:
            log.exception("turn failed")
            yield [*chat, {"role": "assistant", "content": ERROR_MESSAGE}], history, session, ""
            return

        chat = [
            *chat,
            {"role": "assistant", "content": turn.answer},
            {"role": "assistant", "content": format_trace(turn.trace), "metadata": {"title": "How I answered", "status": "done"}},
        ]
        history = [*history, {"role": "user", "content": message}, {"role": "assistant", "content": turn.answer}]
        yield chat, history[-HISTORY_MESSAGES:], session, ""

    def switch_persona(persona_id: str):
        persona = personas[persona_id]
        examples = [gr.update(value=q) for q in persona.example_questions]
        return persona_card(persona), *examples, [], []

    with gr.Blocks(title="Gspänli", analytics_enabled=False) as demo:
        gr.Markdown(ONBOARDING)
        persona = gr.Radio(
            choices=[(p.name, p.id) for p in personas.values()], value=first.id, label="Personality"
        )
        card = gr.Markdown(persona_card(first), elem_classes="persona-card")
        with gr.Row():
            example_buttons = [gr.Button(q, size="sm") for q in first.example_questions]
        chatbot = gr.Chatbot(
            label="Chat",
            height=520,
            placeholder="Pick a personality, then ask a question or click an example.",
        )
        with gr.Row():
            box = gr.Textbox(
                placeholder="Ask in English, French, German or Italian",
                show_label=False,
                scale=8,
                max_lines=4,
                autofocus=True,
            )
            ask = gr.Button("Ask", variant="primary", scale=1)
        new_chat = gr.Button("New conversation", size="sm")

        history = gr.State([])
        session = gr.State({})
        outputs: list[Any] = [chatbot, history, session, box]

        for trigger in (ask.click, box.submit):
            trigger(respond, [box, persona, chatbot, history, session], outputs, concurrency_limit=8)
        for button in example_buttons:
            button.click(respond, [button, persona, chatbot, history, session], outputs, concurrency_limit=8)
        persona.change(switch_persona, persona, [card, *example_buttons, chatbot, history])
        new_chat.click(lambda: ([], []), None, [chatbot, history])
    return demo


def theme() -> gr.themes.Base:
    return gr.themes.Soft(primary_hue="red", neutral_hue="slate")


def create_app():
    """FastAPI app with the Gradio UI mounted at /, used by modal_app.py."""
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    logging.basicConfig(level=logging.INFO)
    demo = build_demo(Runtime(Settings.from_env()))
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/", theme=theme(), css=CSS, footer_links=[])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_demo(Runtime(Settings.from_env())).launch(theme=theme(), css=CSS)
