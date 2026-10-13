"""Static pieces of the UI: theme, CSS, header, status pill, guided tour."""

from __future__ import annotations

import json
from html import escape

import gradio as gr

from .personas import Persona

REPO_URL = "https://github.com/vincvloo/gspaenli"
MCP_URL = "https://github.com/dantevier/swiss-ai-week-mcp"
DRIVER_JS = "https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.js.iife.js"
DRIVER_CSS = "https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.css"

LOGO_SVG = """<svg class="gs-logo" viewBox="0 0 40 40" aria-hidden="true">
<path d="M8 4h24a6 6 0 0 1 6 6v16a6 6 0 0 1-6 6H18l-8 6v-6H8a6 6 0 0 1-6-6V10a6 6 0 0 1 6-6z" fill="#DA291C"/>
<path d="M17 10h6v5h5v6h-5v5h-6v-5h-5v-6h5z" fill="#fff"/></svg>"""


def theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue="red",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        radius_size="lg",
    ).set(
        button_primary_background_fill="#DA291C",
        button_primary_background_fill_hover="#b71f15",
        button_primary_background_fill_dark="#DA291C",
        button_primary_background_fill_hover_dark="#ef4444",
        button_primary_text_color="#ffffff",
    )


def header_html() -> str:
    return f"""
<div class="gs-header">
  <div class="gs-brand">{LOGO_SVG}
    <div><h1>Gspänli</h1><p>Swiss answers, your choice of personality.</p></div>
  </div>
  <nav class="gs-links">
    <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a>
    <a href="{REPO_URL}#how-it-works" target="_blank" rel="noopener">How it works</a>
  </nav>
</div>
<p class="gs-lead">Ask about life in Switzerland in English, French, German or Italian. Every answer
comes from one lookup in official Swiss data and shows its source.</p>"""


STATUS_TEXT = {
    "ready": ("ready", "Model ready"),
    "waking": ("waking", "Model waking up, about 2 min"),
    "asleep": ("asleep", "Model asleep, first answer about 2 min"),
}


def status_html(state: str, ready_until_ms: int = 0) -> str:
    """ready_until_ms lets the browser flip "ready" to "asleep" on its own (see tour_head),
    so no timer has to call the server, which would keep the app container awake."""
    css_class, text = STATUS_TEXT[state]
    expires = f' data-ready-until="{ready_until_ms}"' if state == "ready" and ready_until_ms else ""
    return (
        f'<span class="gs-pill gs-pill-{css_class}"{expires}><span class="gs-dot"></span>'
        f"{escape(text)}</span>"
    )


def persona_label(persona: Persona) -> str:
    return f"{persona.avatar}  {persona.name}"


def placeholder_md(persona: Persona) -> str:
    return (
        f"<div class='gs-empty'><div class='gs-empty-avatar'>{persona.avatar}</div>"
        f"<h3>{escape(persona.name)}</h3><p>{escape(persona.tagline)}</p>"
        "<p class='gs-empty-hint'>Try an example below, or ask your own question.</p></div>"
    )


AVATAR_COLORS = ["#DA291C", "#1f4e79", "#2f7d4f", "#7a4fb3"]


def avatar_png(initial: str, color: str, path: str) -> None:
    """Round avatar with the persona's initial. PNG because Gradio serves SVG as a download
    (SVG can carry scripts), so browsers would not render it."""
    from PIL import Image, ImageDraw, ImageFont

    size = 128
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((0, 0, size - 1, size - 1), fill=color)
    font = ImageFont.load_default(size=64)
    draw.text((size / 2, size / 2), initial, fill="white", font=font, anchor="mm")
    img.save(path)


def persona_css(personas: list[Persona]) -> str:
    """Taglines under each persona radio choice, so the radio renders as cards."""
    rules = []
    for i, persona in enumerate(personas, start=1):
        rules.append(
            f"#gs-personas label:nth-of-type({i})::after {{ content: {json.dumps(persona.tagline)}; }}"
        )
    return "\n".join(rules)


BASE_CSS = """
.gradio-container { max-width: 960px !important; margin: 0 auto !important; }
footer:not(.driver-popover-footer) { display: none !important; }
/* Outside @media on purpose: Gradio scopes @media rules under .contain, which .main wraps. */
.main.fillable { padding-left: clamp(10px, 4vw, 32px) !important; padding-right: clamp(10px, 4vw, 32px) !important; }

.gs-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
.gs-brand { display: flex; align-items: center; gap: 12px; }
.gs-brand h1 { margin: 0; font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; }
.gs-brand p { margin: 0; color: var(--body-text-color-subdued); }
.gs-logo { width: 44px; height: 44px; flex: none; }
.gs-links { display: flex; gap: 16px; }
.gs-links a { color: var(--body-text-color-subdued); text-decoration: none; font-weight: 600; font-size: 0.9rem; white-space: nowrap; }
.gs-links a:hover { color: #DA291C; }
.gs-lead { margin: 8px 0 0; color: var(--body-text-color-subdued); }

#gs-toolbar { align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
#gs-toolbar > div { flex: 0 1 auto !important; min-width: 0 !important; }
#gs-status { width: auto !important; flex: 0 1 auto !important; }
#gs-tour-btn { flex: none !important; min-width: 0 !important; width: auto !important; }
.gs-pill { display: inline-flex; align-items: center; gap: 8px; padding: 4px 12px; border-radius: 999px; white-space: nowrap;
  font-size: 0.85rem; font-weight: 600; border: 1px solid var(--border-color-primary); }
.gs-dot { width: 8px; height: 8px; border-radius: 50%; background: #94a3b8; flex: none; }
.gs-pill-ready .gs-dot { background: #16a34a; }
.gs-pill-waking .gs-dot { background: #f59e0b; animation: gs-pulse 1.2s ease-in-out infinite; }
@keyframes gs-pulse { 50% { opacity: 0.3; } }

#gs-personas { background: none !important; border: none !important; padding: 0 !important; }
#gs-personas .wrap { display: grid !important; grid-template-columns: repeat(3, 1fr); gap: 10px; }
#gs-personas label { display: flex !important; flex-direction: column; align-items: flex-start !important;
  gap: 4px; padding: 14px 16px !important; border-radius: 14px !important; cursor: pointer;
  border: 2px solid var(--border-color-primary) !important; background: var(--background-fill-primary) !important;
  font-weight: 700; font-size: 1rem; transition: border-color .15s, transform .15s; }
#gs-personas label:hover { border-color: #f1948c !important; transform: translateY(-1px); }
#gs-personas label.selected, #gs-personas label:has(input:checked) { border-color: #DA291C !important;
  background: var(--color-accent-soft) !important; }
#gs-personas label, #gs-personas label * , #gs-personas label.selected, #gs-personas label.selected * {
  color: var(--body-text-color) !important; }
#gs-personas label::after { font-weight: 400; font-size: 0.85rem; color: var(--body-text-color-subdued); }
#gs-personas input { position: absolute; opacity: 0; pointer-events: none; }

#gs-chat { border-radius: 16px; }
/* No user avatar is used, so do not reserve space for one; let answers use the full width. */
#gs-chat .message-row.bot-row { max-width: 100% !important; margin-right: 0 !important; padding-right: 0 !important; }
#gs-chat .message-row.bot-row .flex-wrap { flex: 1 1 auto; min-width: 0; }
#gs-chat .message-row.bot-row .message { max-width: 100% !important; }
/* Sources are long URLs; without this they widen the whole page on phones. */
#gs-chat .message-content, #gs-chat .message-content a { overflow-wrap: anywhere; word-break: break-word; }
.gs-empty { text-align: center; padding: 24px 12px; }
.gs-empty-avatar { font-size: 2.6rem; }
.gs-empty h3 { margin: 6px 0 2px; }
.gs-empty p { margin: 4px 0; color: var(--body-text-color-subdued); }
.gs-empty-hint { font-size: 0.85rem; }

#gs-examples { gap: 8px; flex-wrap: wrap; }
#gs-examples button { flex: 1 1 200px; font-weight: 500; text-align: left; justify-content: flex-start;
  white-space: normal; border-radius: 12px; }

#gs-input { align-items: stretch; gap: 8px; }
#gs-ask { min-width: 90px !important; }
.gs-footnote { font-size: 0.8rem; color: var(--body-text-color-subdued); text-align: center; }
.gs-footnote a { color: inherit; }

@media (max-width: 640px) {
  .gs-brand h1 { font-size: 1.35rem; }
  .gs-logo { width: 36px; height: 36px; }
  .gs-lead { font-size: 0.9rem; }
  #gs-personas .wrap { grid-template-columns: 1fr; gap: 6px; }
  #gs-personas label { padding: 10px 12px !important; }
  #gs-examples button { flex-basis: 100%; }
  .gs-pill { font-size: 0.75rem; }
}

.driver-popover { border-radius: 14px; max-width: 320px; }
.driver-popover-title { font-weight: 800; }
.driver-popover-footer button { border-radius: 8px; }
.driver-popover-next-btn { background: #DA291C !important; color: #fff !important; text-shadow: none !important; border: none !important; }
.gs-skip-btn { margin-right: auto; background: none !important; border: none !important; color: #64748b !important;
  text-shadow: none !important; cursor: pointer; }
"""

TOUR_STEPS = [
    ("#gs-header", "Grüezi! This is Gspänli",
     "A small demo: ask about life in Switzerland and get an answer grounded in official Swiss data."),
    ("#gs-personas", "1. Pick a personality",
     "Same facts, three voices: a warm guide for newcomers, a precise federal clerk, or your rule-loving neighbour."),
    ("#gs-examples", "2. Start with an example",
     "Each personality has three example questions. Click one to try it."),
    ("#gs-input", "3. Ask in your language",
     "English, French, German or Italian. You get the answer in your language, even when the source is in another one."),
    ("#gs-chat", "4. See how I answered",
     "Under every answer, open <b>How I answered</b>: the tool I used, its arguments, my assumptions and the official source."),
    ("#gs-status", "5. The model naps",
     "The open model Apertus runs on a GPU that sleeps when idle. The first answer can take about 2 minutes, the next ones a few seconds."),
]


def tour_head() -> str:
    steps = [
        {"element": sel, "popover": {"title": title, "description": desc}} for sel, title, desc in TOUR_STEPS
    ]
    return f"""
<link rel="stylesheet" href="{DRIVER_CSS}">
<script src="{DRIVER_JS}"></script>
<script>
(function () {{
  const KEY = "gspaenli-tour-done";
  const steps = {json.dumps(steps)};
  function store(set) {{
    try {{ return set ? localStorage.setItem(KEY, "1") : localStorage.getItem(KEY); }} catch (e) {{ return null; }}
  }}
  window.gspaenliTour = function () {{
    if (!window.driver || !window.driver.js) return;
    const tour = window.driver.js.driver({{
      showProgress: true,
      progressText: "{{{{current}}}} of {{{{total}}}}",
      nextBtnText: "Next",
      prevBtnText: "Back",
      doneBtnText: "Start chatting",
      steps: steps.filter((s) => document.querySelector(s.element)),
      onPopoverRender: (popover) => {{
        const skip = document.createElement("button");
        skip.innerText = "Skip";
        skip.className = "gs-skip-btn";
        skip.addEventListener("click", () => tour.destroy());
        popover.footer.prepend(skip);
      }},
      onDestroyed: () => store(true),
    }});
    tour.drive();
  }};
  // The GPU scales down when idle: expire a "ready" pill locally, without calling the server.
  setInterval(function () {{
    const pill = document.querySelector(".gs-pill[data-ready-until]");
    if (pill && Date.now() > Number(pill.dataset.readyUntil)) {{
      pill.className = "gs-pill gs-pill-asleep";
      pill.removeAttribute("data-ready-until");
      pill.lastChild.textContent = {json.dumps(STATUS_TEXT["asleep"][1])};
    }}
  }}, 15000);
  // First visit: start the tour once the Gradio app has rendered.
  let tries = 0;
  const timer = setInterval(function () {{
    tries += 1;
    if (document.querySelector("#gs-personas") && window.driver) {{
      clearInterval(timer);
      if (!store(false)) setTimeout(window.gspaenliTour, 400);
    }} else if (tries > 100) {{
      clearInterval(timer);
    }}
  }}, 100);
}})();
</script>"""
