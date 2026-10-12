# Handover: Gspänli (POC)

**Name:** Gspänli (Swiss German for buddy). Repo and Modal app slug: `gspaenli`. Subtitle: "Swiss answers, your choice of personality."

You are taking over a small POC. Read this whole file before doing anything.

## How to work (mandatory order)

1. **Explore.** Clone the MCP repo (see below), read its README, `pyproject.toml`, `src/mcp_swiss_info/main.py` and `server.py`, and list the registered tools. Check current Modal docs for vLLM serving and web endpoints. Do not rely on memory for Modal APIs, GPU names or model IDs: verify them.
2. **Draft plan.** Write `docs/PLAN.md`: architecture, file layout, milestones, risks, test approach.
3. **Critique your own plan.** Add a section `## Critique` in the same file. Attack it honestly: what breaks, what is over-engineered for one evening, what is untested, what costs Modal credit unnecessarily, where tool calling will fail. At least 8 concrete points.
4. **Final plan.** Rewrite the plan into `## Final plan`, with one PR per milestone, each PR with scope, files touched, and a "done when" check.
5. **Stop and wait** for my approval of the final plan. Do not write application code before that.
6. **Implement PR by PR.** One branch and one PR per milestone. Each PR: small, runs, has a short description (what, why, how to test), and is merged before starting the next. Commit often with clear messages.

## Goal

A public web demo, usable with **no install**, where a user picks a **personality** (prompt-based) and chats about life in Switzerland. Answers are grounded in our existing **Swiss MCP server**. The LLM is **open source**, served on **Modal**, default **Apertus** (the Swiss open model), switchable to another open model by a parameter.

This is the project to show learning from a training on finetuning personalities on Modal. We deliberately chose **prompting over finetuning** (too little data and training budget for a finetune to beat a good prompt). The README must explain that decision in a short, defensible paragraph. A tiny finetune experiment is optional and last.

**Multilingual is a core feature.** Users ask in English, French, German or Italian and get the answer in their language, even when the source is in another one. README claim: "Ask in your language, grounded in official sources in theirs."

Time budget: one evening (~4h). Modal credit is limited. Prefer simple and working over complete.

## Context: the Swiss MCP (do not modify it)

- Repo: https://github.com/dantevier/swiss-ai-week-mcp (public, MIT). Use tag `v1.0`.
- Python >= 3.11, `uv`, FastMCP v2. Transport **stdio** (also supports `--transport sse --port`).
- Start: `uv run --frozen python -m mcp_swiss_info.main`
- No credentials needed. Ships ~65 MB of data in the repo.
- Tools include: `swiss_health_insurance_premiums`, `swiss_residence_permit_guidance`, `swiss_housing_info`, `swiss_reference_interest_rate`, `get_driving_licence_exchange_info`, `swiss_school_holidays`, `compute_swiss_tax`, `company_info`, `swiss_geo_search`, `swiss_geo_context`, `swiss_weather_forecast`, `swiss_weather_observations`, `swiss_federal_political_rights`, `swiss_import_parcel_vat`, `bfs_population`, `search_knowledge`, `get_source`, `opendata_search_datasets`, `opendata_dataset`.
- **Exclude** from the agent: `crawl_federal_sources`, `crawl_cantonal_sources`, `crawl_municipal_sources`, `open_dashboard`, `source_status` (side effects or not useful for chat). Use an allow-list, not a deny-list.

Preferred integration: vendor it as a git submodule or clone at image build time in Modal, run it as a stdio subprocess inside the app container, talk to it with the official MCP Python client. Verify this works locally before deploying.

## Target architecture

```
Browser -> Gradio app (Modal, CPU, public URL)
             |- persona file (system prompt + preferred tools)
             |- MCP client -> swiss MCP (stdio subprocess, same container)
             |- OpenAI-compatible client -> LLM server (Modal, GPU, vLLM)
```

- **LLM server**: separate Modal app, vLLM, OpenAI-compatible API, scale to zero, protected by a simple bearer token stored as a Modal secret.
- **Model is a parameter**: `MODEL_ID` env/config. Default `swiss-ai/Apertus-8B-Instruct-2509` (verify the exact ID on Hugging Face). One alternative documented and tested, e.g. a Qwen or Mistral instruct model of similar size. GPU type also configurable. Choose the cheapest GPU that fits an 8B model in bf16.
- **App** reads `LLM_BASE_URL`, `LLM_API_KEY`, `MODEL_ID` from Modal secrets/env, so it can also point at any OpenAI-compatible endpoint (e.g. Public AI) without code change.

## Tool flow (decided)

Do **not** depend on native tool calling. Use a two-step flow that works with any model:

1. **Route**: model gets the question, the persona's preferred tools and the allow-listed tool schemas, and returns strict JSON: `{"tool": "<name>|none", "arguments": {...}}`. Validate it; on parse failure retry once, then fall back to `search_knowledge` with the raw question.
2. **Call** the MCP tool, truncate the result to a sane size.
3. **Answer**: model answers in persona using only the tool result, cites the source, says clearly when data is missing.

Max one tool call per turn for the POC (document multi-step as a next step).

## Personas

One YAML file each in `personas/`. Adding a persona = adding a file. Fields: `id`, `name`, `tagline`, `system_prompt`, `preferred_tools`, `example_questions` (3), `language_hint`.

1. **Newcomer guide**: warm, step-by-step, assumes the user just arrived. Favours permits, health insurance, housing, driving licence, school holidays.
2. **Fedlex clerk**: dry, precise, cites the source, refuses to guess, short answers. Favours `search_knowledge`, `get_source`, political rights, tax.
3. **Bünzli neighbour**: Swiss humour, rule-loving, mildly passive-aggressive but helpful. Favours waste/housing rules, reference interest rate, weather, holidays.

All personas: never invent figures; if the tool returned nothing, say so in character.

**Language rules (all personas):**
- Reply in the language of the user's question (EN, FR, DE, IT minimum), whatever the language of the source.
- Routing must work for questions in all four languages; tool names and descriptions stay in English.
- Keep persona character in every language; Swiss German output is a bonus, not a promise.

## UI (Gradio)

- Persona selector with tagline and its 3 example questions as clickable prompts.
- Chat.
- A collapsible **"How I answered"** panel per answer: tool chosen, arguments, source/URL, latency, model ID. This makes the MCP visible and is the core of the demo.
- A clear notice when the LLM is cold-starting.
- A short onboarding text at the top: what this is, what data it uses, that it is a POC.

## Repo layout (suggestion, challenge it in your critique)

```
personas/*.yaml
app/            # gradio app, orchestration, mcp client, llm client
llm_server/     # modal vLLM app
evals/questions.yaml   # ~10 questions with expected tool and language
tests/
docs/PLAN.md
README.md
.env.example
```

## Milestones (rework in your final plan)

1. Skeleton, personas, config, local MCP client listing tools.
2. vLLM on Modal with `MODEL_ID` param, bearer auth, curl test.
3. Orchestration (route, call, answer) runnable from CLI against Modal LLM.
4. Gradio UI deployed on Modal with public URL.
5. README with live link, architecture, quickstart, "why prompting not finetuning", known limits.
6. Optional: run `evals/questions.yaml` on two models (Apertus vs alternative), small results table. Optional tiny LoRA note.

## Constraints

- Secrets only via Modal secrets / `.env` (gitignored). Never commit tokens. Ask me to run `modal secret create` myself.
- Keep GPU costs low: scale to zero, short idle timeout, no 70B models, no long-running test loops on GPU.
- Python 3.11+, `uv`. Pin dependencies.
- `evals/questions.yaml` includes at least one question per language (EN, FR, DE, IT) and one where the source language differs from the question language.
- Tests for: persona loading, router JSON parsing and fallback, tool allow-list, reply-language instruction present in every prompt. Mock the LLM in tests.
- Writing style for README and UI text: direct, concise, no em-dashes, no marketing fluff.

## What I will provide

- Modal account access (I run `modal setup` and create secrets when you ask).
- Hugging Face token as a Modal secret.
- An empty GitHub repo for this project.

## Definition of done

A public Modal URL where someone picks a persona, asks "What is the minimum health insurance premium in Lausanne for an adult?" and gets an in-character answer with the tool and source shown, and the same question asked in French gets the answer in French, plus a README a stranger understands in 2 minutes.
