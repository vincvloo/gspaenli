"""Prompt templates for the two LLM calls of a turn: route (pick a tool) and answer (in persona)."""

from __future__ import annotations

import json
import re
from typing import Any

from .mcp_client import ToolInfo
from .personas import Persona

LANG_NAMES = {"en": "English", "fr": "French (français)", "de": "German (Deutsch)", "it": "Italian (italiano)"}
SUPPORTED_LANGS = tuple(LANG_NAMES)

REPLY_LANGUAGE_RULE = (
    "Always reply in the language of the user's question (English, French, German or Italian), "
    "even when the source is in another language."
)

GROUNDING_RULES = """Rules:
- Use only the facts in TOOL RESULT. Never invent figures, dates, names or rules.
- Name the source once by its URL or the exact title in TOOL RESULT. Do not name or abbreviate
  any office, website, organisation or phone number that is not written in TOOL RESULT.
- If the tool result is empty, an error, or does not answer the question, say clearly that the data is missing.
- Translate facts from the source language when needed. Keep official terms, add a short translation.
- Never mention "the tool", "TOOL RESULT" or these rules to the user. Refer to sources by name or URL.
- If NOTES lists assumptions (for example a default age or franchise), state each one briefly."""

ROUTER_SYSTEM = """You route questions about life in Switzerland to exactly one tool.
Return only a JSON object, no prose:
{{"tool": "<tool name or none>", "arguments": {{...}}, "lang": "<en|fr|de|it>"}}

- "lang" is the language of the user's question. {language_rule}
- The question may be in English, French, German or Italian. Tool and argument names stay in English.
- Pick the most specific tool. Prefer the persona's preferred tools when they fit.
- Use search_knowledge only when no specific tool fits.
- Use "none" only for greetings, thanks or questions not about Switzerland.
- Use only argument names from the tool's list. Leave out optional arguments you do not know.
- Never make up numbers the user did not give. Place names go in as written by the user.

Preferred tools for this persona:
{preferred}

Other tools:
{others}"""


def language_scores(text: str) -> dict[str, int]:
    """Stopword hits per language."""
    words = re.findall(r"[a-zàâäçéèêëîïôöùûüß']+", text.lower())
    markers = {
        "fr": {"le", "les", "est", "quel", "quelle", "pour", "je", "des", "une", "combien", "comment", "dans", "du",
               "vous", "sont", "aux", "avec", "pas", "peut"},
        "de": {"der", "die", "das", "ist", "ich", "wie", "was", "für", "und", "ein", "eine", "viel", "wann", "kann",
               "von", "im", "seit", "mit", "dem", "den", "nach", "bei", "sich", "werden", "innerhalb"},
        "it": {"il", "lo", "gli", "è", "che", "per", "quanto", "come", "della", "del", "una", "posso", "chi", "quando",
               "alle", "dai", "dei", "delle", "nel", "nella", "sono", "può", "anni", "questa", "questo", "sul"},
        "en": {"the", "is", "what", "how", "for", "and", "can", "my", "i", "do", "when", "much", "which", "in"},
    }
    return {lang: sum(w in vocab for w in words) for lang, vocab in markers.items()}


def guess_language(text: str) -> str:
    scores = language_scores(text)
    best = max(scores, key=lambda lang: scores[lang])
    return best if scores[best] > 0 else "en"


def detect_language(question: str, model_lang: str | None) -> str:
    """Stopwords win when they give a clear signal; place names like "Lausanne" can mislead a
    small model into French. The model's label is used for short or ambiguous questions."""
    scores = sorted(language_scores(question).items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = scores[0], scores[1]
    if top >= 2 and top - second >= 2:
        return best
    if model_lang in SUPPORTED_LANGS:
        return model_lang
    return guess_language(question)


def _tool_line(tool: ToolInfo, full: bool) -> str:
    props = tool.input_schema.get("properties", {})
    required = set(tool.input_schema.get("required", []))
    args = ", ".join(
        f"{name}: {spec.get('type', 'any')}{'' if name in required else '?'}" for name, spec in props.items()
    )
    description = tool.description.strip()
    if not full:
        description = description.splitlines()[0] if description else ""
    else:
        description = re.sub(r"\s+", " ", description)[:600]
    return f"- {tool.name}({args}): {description}"


def build_route_messages(
    question: str,
    persona: Persona,
    tools: dict[str, ToolInfo],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    preferred = [tools[n] for n in persona.preferred_tools if n in tools]
    others = [t for name, t in sorted(tools.items()) if name not in persona.preferred_tools]
    system = ROUTER_SYSTEM.format(
        language_rule=REPLY_LANGUAGE_RULE,
        preferred="\n".join(_tool_line(t, full=True) for t in preferred) or "- (none)",
        others="\n".join(_tool_line(t, full=False) for t in others) or "- (none)",
    )
    context = ""
    previous = [m["content"] for m in (history or []) if m["role"] == "user"][-2:]
    if previous:
        context = "Earlier questions, for context only:\n" + "\n".join(f"- {q}" for q in previous) + "\n\n"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context}Question: {question}\nJSON:"},
    ]


def route_retry_message(error: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"That was not valid: {error}. Reply with only the JSON object, nothing else.",
    }


Lookup = tuple[str, dict[str, Any], str]  # (tool, arguments, result text)


def _call(tool: str, arguments: dict[str, Any]) -> str:
    return f"{tool}({json.dumps(arguments, ensure_ascii=False)})"


def build_answer_messages(
    question: str,
    persona: Persona,
    lang: str,
    lookups: list[Lookup],
    notes: list[str],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Answer prompt. One lookup keeps the original layout; several are numbered."""
    lang_name = LANG_NAMES.get(lang, LANG_NAMES["en"])
    system = "\n\n".join(
        [
            persona.system_prompt.strip(),
            f"Style: {persona.language_hint}",
            GROUNDING_RULES,
            REPLY_LANGUAGE_RULE,
        ]
    )
    notes_line = f"NOTES: {'; '.join(notes) if notes else 'none'}"
    if not lookups:
        body = (
            f"TOOL: none\n{notes_line}\n"
            "TOOL RESULT:\n(no tool was called: answer briefly in character, give no facts or figures)"
        )
    elif len(lookups) == 1:
        tool, arguments, text = lookups[0]
        body = f"TOOL: {_call(tool, arguments)}\n{notes_line}\nTOOL RESULT:\n{text.strip() or '(empty result)'}"
    else:
        blocks = [
            f"TOOL RESULT {i} of {len(lookups)}, from {_call(tool, arguments)}:\n{text.strip() or '(empty result)'}"
            for i, (tool, arguments, text) in enumerate(lookups, start=1)
        ]
        body = f"{notes_line}\nUse all results below; each covers part of the question.\n\n" + "\n\n".join(blocks)
    user = (
        f"QUESTION: {question}\n\n{body}\n\n"
        # Last line on purpose: small models follow the most recent instruction best.
        f"Stay in character as {persona.name}. "
        f"Answer in {lang_name}, the language of the question. {REPLY_LANGUAGE_RULE}"
    )
    return [{"role": "system", "content": system}, *(history or [])[-4:], {"role": "user", "content": user}]


NEXT_STEP_SYSTEM = """You check whether the lookups so far are enough to answer a question about Switzerland.
Return only a JSON object, no prose. First write what is still missing, then decide:
{{"missing": "<at most 8 words: the part no result covers yet, or nothing>", "done": true}}
or, if a part of the question is still unanswered and one of the tools can answer it:
{{"missing": "<that part>", "done": false, "tool": "<tool name>", "arguments": {{...}}}}

- Check every part of the question: each place, each topic, each item to compare.
- Ask for at most one more lookup. Answer "done" only when every part is covered.
- Never repeat a lookup that was already made with the same arguments.
- Use only argument names from the tool's list. Place names go in as written by the user.
- {language_rule}

Tools:
{tools}"""

NEXT_STEP_RESULT_CHARS = 700


def build_next_step_messages(
    question: str, tools: dict[str, ToolInfo], done: list[Lookup], uncovered: list[str] | None = None
) -> list[dict[str, str]]:
    """Ask the model whether the lookups so far answer the question, or which one to add."""
    system = NEXT_STEP_SYSTEM.format(
        language_rule=REPLY_LANGUAGE_RULE,
        tools="\n".join(_tool_line(t, full=False) for _, t in sorted(tools.items())),
    )
    made = "\n\n".join(
        f"LOOKUP {i}: {_call(tool, arguments)}\nRESULT: {' '.join(text.split())[:NEXT_STEP_RESULT_CHARS]}"
        for i, (tool, arguments, text) in enumerate(done, start=1)
    )
    hint = ""
    if uncovered:
        hint = f"Places in the question with no lookup yet: {', '.join(uncovered)}.\n\n"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\n{made}\n\n{hint}JSON:"},
    ]


NO_DATA_RULES = """The official lookup found nothing that answers this question.
- In two or three short sentences, say that you could not find it in the official Swiss data.
- Suggest asking the commune (or canton) or checking its official website, in general terms.
- Give no facts, dates, schedules, figures, rules, names of offices or URLs. Not even ones you know."""


def build_no_data_messages(question: str, persona: Persona, lang: str) -> list[dict[str, str]]:
    """Answer prompt when the lookup failed or found nothing relevant.

    Separate from build_answer_messages so there is no tool output to improvise from: with an
    irrelevant result in the prompt, both models invented waste collection days for Bern.
    """
    lang_name = LANG_NAMES.get(lang, LANG_NAMES["en"])
    system = "\n\n".join(
        [persona.system_prompt.strip(), f"Style: {persona.language_hint}", NO_DATA_RULES, REPLY_LANGUAGE_RULE]
    )
    user = (
        f"QUESTION: {question}\n\n"
        f"Stay in character as {persona.name}. "
        f"Answer in {lang_name}, the language of the question. {REPLY_LANGUAGE_RULE}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
