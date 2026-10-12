# Gspänli: plan

Status: waiting for approval. No application code yet.

## Findings from exploration (verified 2026-10-12)

MCP (`dantevier/swiss-ai-week-mcp`, tag `v1.0` = `be988a0e02d9f66163f4f9bea6f786be3209a599`):

- Runs over stdio with `uv run --frozen python -m mcp_swiss_info.main`. Locked FastMCP is 4.0.9 (not v2 as the handover says). Works with the official `mcp` Python client (tested with `mcp` 1.x).
- First start took ~25 s (it builds `var/swissproject.sqlite3` in its lifespan hook), later starts ~4 s. Tool calls on local data take <1 s, network tools (geo, weather, holidays) ~1.5 s.
- 23 tools registered. **`compute_swiss_tax` is not registered at v1.0** (file exists, the import is commented out as a template demo). The handover list is otherwise correct.
- **`swiss_health_insurance_premiums` requires `age`, `municipality_code` (BFS number, not postal code) and `franchise`** (300/500/1000/1500/2000/2500). A model cannot know that Lausanne is 5586. `swiss_geo_search` returns names and coordinates but no BFS number; `swiss_geo_context(lat, lon)` does return it. So the definition-of-done question needs either two MCP calls or a local lookup.
- The MCP repo ships `data/swiss_places_premiums_2026.sqlite` with tables `communes` and `place_aliases` (name, normalized alias, BFS code). We can read it (read-only, no modification) to resolve "Lausanne" / "Losanna" to 5586 locally in <1 ms.
- Verified chain: `swiss_health_insurance_premiums(age=35, municipality_code=5586, franchise=300)` returns 531.70 CHF (no accident cover, Sanitas TelMed) with source URLs.
- `search_knowledge` uses keyword fallback without `OPENAI_API_KEY` and can return noisy raw JSON passages (5 kB for 2 results). Must be truncated and cleaned.
- All tool schemas together are ~5.8 kB of JSON (~1.5k tokens). Fine to put in the router prompt.

Model and Modal:

- `swiss-ai/Apertus-8B-Instruct-2509` exists, Apache-2.0, 65k context, bf16, needs transformers >= 4.56. Recommended sampling: temperature 0.8, top_p 0.9.
- vLLM `v0.21.0` (pinned by Modal's current `vllm_inference` example) contains `vllm/model_executor/models/apertus.py`, so native support. Latest vLLM is 0.31.0; we pin 0.21.0 because it is the version Modal tests.
- Modal GPU prices: T4 $0.59/h (16 GB, too small for 8B bf16), **L4 $0.80/h (24 GB)**, A10 $1.10/h (24 GB), L40S $1.95/h. L4 is the cheapest that fits 8B bf16 (~16 GB weights + KV cache at `max_model_len` 8192).
- Modal patterns: `modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")`, HF cache on a `modal.Volume`, `@modal.web_server(port)` or `@app.server(...)` for vLLM, `@modal.asgi_app()` + `gradio.routes.mount_gradio_app` with `max_containers=1` for Gradio (sticky sessions). vLLM `--api-key` gives bearer auth natively.
- Latest packages: modal 1.6.1, gradio 6.30.0, mcp 2.3.0, openai 3.29.0. Pin these (re-check at PR1).

## Draft plan

### Architecture

```
Browser -> Modal app "gspaenli" (CPU, asgi: FastAPI + Gradio, public URL)
             |- personas/*.yaml
             |- orchestrator: route -> resolve args -> MCP call -> answer
             |- MCP client -> swiss MCP (stdio subprocess, one per container, started at container start)
             |- OpenAI client -> Modal app "gspaenli-llm" (GPU, vLLM, --api-key bearer)
```

### File layout

```
gspaenli/            # python package (importable, testable)
  config.py          # env: LLM_BASE_URL, LLM_API_KEY, MODEL_ID, MCP_DIR
  personas.py        # load + validate personas/*.yaml
  mcp_client.py      # stdio session, allow-list, call + truncate
  llm.py             # OpenAI-compatible client wrapper
  router.py          # route prompt, JSON parse, retry, fallback
  resolver.py        # municipality name -> BFS code, defaults for age/franchise
  orchestrator.py    # one turn: route, resolve, call, answer, trace
  prompts.py         # prompt templates incl. reply-language rule
  cli.py             # `uv run gspaenli ask "..." --persona newcomer`
  ui.py              # Gradio Blocks
modal_app.py         # Modal CPU app serving ui.py
llm_server/serve.py  # Modal GPU app, vLLM
personas/*.yaml
evals/questions.yaml, evals/run.py
tests/
scripts/fetch_mcp.sh # clone MCP at the pinned commit into vendor/ (gitignored)
docs/PLAN.md
README.md, .env.example, pyproject.toml, uv.lock
```

### Milestones

1. Skeleton, personas, config, MCP client that lists allow-listed tools locally.
2. vLLM server on Modal, `MODEL_ID` and `GPU` params, bearer auth, curl test.
3. Orchestrator + CLI against the Modal LLM.
4. Gradio UI on Modal, public URL.
5. README.
6. Optional: evals Apertus vs alternative.
7. Optional: voice (ElevenLabs) and a static GitHub Pages front end.

### Risks

- Router JSON fails on an 8B model, especially for FR/DE/IT questions.
- Premium tool needs a BFS code the model cannot know.
- Cold start of the GPU (image pull + 16 GB weights + CUDA graphs) could be 2 to 4 minutes; Gradio requests may time out.
- Public URL lets anyone burn GPU credit.
- MCP subprocess dies or blocks the event loop.

### Test approach

- Unit tests with a fake LLM (scripted responses) and a fake MCP session: persona loading, router parsing/retry/fallback, allow-list, language rule in every prompt, resolver.
- One local integration test (marked `integration`) that starts the real MCP over stdio and calls 2 offline tools.
- Manual GPU tests only with `curl` and a handful of CLI questions. Evals run once per model.

## Critique

1. **One tool call per turn cannot answer the definition-of-done question.** The premium tool needs a BFS code. Letting the model guess the code will produce confident wrong numbers. Fix: the router outputs `municipality` (a name) and the app resolves it to `municipality_code` via the MCP's own `place_aliases` table. This keeps "one MCP tool call" honest and is deterministic. If the name is not found, fall back to `swiss_geo_search` + `swiss_geo_context` (two calls, logged in the trace).
2. **"Minimum premium for an adult" is underspecified.** Age and franchise are required. The answer must state the assumption. Default: age 30 (adult 26+), franchise 300 (default in Switzerland), and say that higher franchises lower the premium. Do not silently pick 2500.
3. **`compute_swiss_tax` does not exist at v1.0.** The Fedlex clerk's preferred tools must not reference it. The allow-list must be validated against `list_tools()` at startup, logging unknown names instead of crashing.
4. **Prompt-only JSON from an 8B model is fragile.** vLLM supports structured outputs (`response_format` with a JSON schema), which would make routing near-perfect, but it breaks the "any OpenAI-compatible endpoint" promise. Fix: send `response_format` when `LLM_SUPPORTS_JSON_SCHEMA=true` (default true for our vLLM), keep the tolerant parser (strip code fences, extract first `{...}`) plus retry plus `search_knowledge` fallback for other endpoints.
5. **Putting all 18 tool schemas in every route prompt wastes tokens and confuses a small model.** Put the persona's preferred tools first with full descriptions, the rest with one-line descriptions. Do not drop them: a newcomer can still ask about weather.
6. **Separate `app/` and `llm_server/` plus a submodule is over-engineered for one evening.** A git submodule adds friction (clone flags, Windows paths) and Modal would re-upload 65 MB. Fix: clone the MCP at the pinned commit during the Modal image build (cached layer) and with `scripts/fetch_mcp.sh` locally. One Python package `gspaenli/`.
7. **MCP cold init of 25 s inside the app container.** Fix: run `uv sync --frozen` and one server start (which builds `var/swissproject.sqlite3`) at image build time, so container start is ~4 s. Start the MCP once per container in the FastAPI lifespan, not per request. Serialize calls through an `asyncio.Lock` (stdio is one pipe; FastMCP handles concurrency but the POC does not need it).
8. **GPU cold start vs Gradio timeouts.** vLLM with CUDA graph capture on L4 can take minutes. Fix: `--enforce-eager` (faster boot, slightly slower decode), HF weights on a Volume, `startup_timeout` 10 min, and the UI first pings `/health` with a short timeout: if not ready, show "The model is waking up (about 2 minutes). Your question is queued." and stream status updates via a generator.
9. **L4 is the cheapest, but slow.** 300 GB/s memory bandwidth with 16 GB weights gives ~15 to 18 tokens/s. Two LLM calls per turn (short route + ~250-token answer) means ~15 to 20 s per turn. A10 is ~2x faster for 1.4x the price. Make GPU a deploy param, default L4, measure once, switch only if the demo feels broken.
10. **Credit burn from a public URL.** Fix: GPU app `max_containers=1`, `scaledown_window` 5 min, `max_tokens` caps (route 200, answer 500), app-side rate limit (per session 20 turns, global 300/day in memory), `max_model_len` 8192, and ask the user to set a Modal workspace budget limit. The LLM endpoint is bearer-protected so only the app can call it.
11. **Language detection is left to the model.** An 8B model may answer in English when the tool result is German. Fix: detect the question language cheaply (router JSON includes `"lang": "en|fr|de|it"`), and put "Answer in {lang_name}" at the end of the answer prompt, after the tool result (recency matters more than the system prompt for small models). Test that the rule is present in every prompt.
12. **Tool results are messy.** `search_knowledge` returned raw CKAN JSON. Fix: truncate to ~3k chars, extract `source_url`/`url` fields into the trace for citations, so the source shown in the UI comes from data, not from the model.
13. **Qwen3 as the alternative has a thinking mode** that spends tokens in `<think>` blocks and breaks JSON. Fix: pass `chat_template_kwargs={"enable_thinking": false}` when the model ID starts with `Qwen/Qwen3`, and strip `<think>` blocks defensively. Verify the exact ID (`Qwen/Qwen3-8B`) on Hugging Face in PR2.
14. **Untested on GPU until late.** Fix: PR2 deploys the LLM server early and is checked with curl, so PR3 is developed against the real model. Keep GPU sessions short and stop the app when idle (`modal app stop` is not needed thanks to scale to zero, but avoid test loops).
15. **Gradio on Modal needs sticky sessions**, so `max_containers=1` for the UI, with `@modal.concurrent(max_inputs=50)`. Fine for a demo, documented as a known limit.

## Final plan

Conventions: branch `pr/<n>-<slug>`, one PR per milestone, squash-merge after the "done when" check passes. Python 3.11+ with `uv`, all dependencies pinned in `pyproject.toml` and `uv.lock`. No secrets in git.

Modal secrets I will ask you to create (you run the commands):

- `huggingface` with `HF_TOKEN`
- `gspaenli-llm` with `LLM_API_KEY` (random string, e.g. `python -c "import secrets;print(secrets.token_urlsafe(32))"`)
- `gspaenli-app` with `LLM_BASE_URL`, `LLM_API_KEY` (same value), `MODEL_ID`

### PR 1: skeleton, personas, MCP client

- Scope: `pyproject.toml`, package skeleton, `config.py`, `personas.py`, 3 persona YAMLs, `mcp_client.py` with allow-list, `scripts/fetch_mcp.sh`, `.env.example`, `.gitignore`, tests.
- Allow-list (18): `swiss_health_insurance_premiums`, `swiss_residence_permit_guidance`, `swiss_housing_info`, `swiss_reference_interest_rate`, `get_driving_licence_exchange_info`, `swiss_school_holidays`, `company_info`, `swiss_geo_search`, `swiss_geo_context`, `swiss_weather_forecast`, `swiss_weather_observations`, `swiss_federal_political_rights`, `swiss_import_parcel_vat`, `bfs_population`, `search_knowledge`, `get_source`, `opendata_search_datasets`, `opendata_dataset`. (`compute_swiss_tax` added automatically if a future tag registers it.)
- Fedlex clerk favours `search_knowledge`, `get_source`, `swiss_federal_political_rights`, `swiss_import_parcel_vat`, `company_info` (no tax tool).
- Files: `pyproject.toml`, `gspaenli/{__init__,config,personas,mcp_client}.py`, `personas/{newcomer,fedlex_clerk,buenzli}.yaml`, `scripts/fetch_mcp.sh`, `tests/test_personas.py`, `tests/test_allowlist.py`, `tests/test_mcp_integration.py`.
- Done when: `uv run pytest` passes; `uv run python -m gspaenli.mcp_client` prints the allow-listed tools and calls the premium tool for 5586.

### PR 2: vLLM server on Modal

- Scope: `llm_server/serve.py`. vLLM 0.21.0, `MODEL_ID` and `GPU` read from env at deploy time (default Apertus, L4), `--api-key` from secret, `--max-model-len 8192`, `--enforce-eager`, HF Volume, `scaledown_window` 300 s, `max_containers=1`, `startup_timeout` 10 min.
- Files: `llm_server/serve.py`, `llm_server/README.md` (deploy + curl).
- Done when: `curl -H "Authorization: Bearer $KEY" .../v1/models` lists Apertus; a chat completion in French answers in French; a call without the token returns 401; cold start time noted.

### PR 3: orchestration + CLI

- Scope: `prompts.py`, `llm.py`, `router.py` (JSON schema output when supported, tolerant parse, 1 retry, fallback to `search_knowledge`), `resolver.py` (BFS lookup from `place_aliases`, defaults age 30 / franchise 300 stated in answer, geo fallback), `orchestrator.py` returning answer + trace (tool, args, source URLs, latency, model), `cli.py`. `evals/questions.yaml` (10 questions, EN/FR/DE/IT, at least one cross-language).
- Files: those above + `tests/test_router.py`, `tests/test_prompts_language.py`, `tests/test_resolver.py`, `tests/test_orchestrator.py` (fake LLM, fake MCP).
- Done when: unit tests pass; `uv run gspaenli ask "What is the minimum health insurance premium in Lausanne for an adult?" --persona newcomer` returns a figure matching the tool, with the source; the French version answers in French.

### PR 4: Gradio UI on Modal

- Scope: `ui.py` (persona selector with tagline and 3 clickable examples, chat, "How I answered" accordion per answer, cold-start notice via `/health` probe, onboarding text), `modal_app.py` (image with MCP cloned at the pinned commit and prewarmed, FastAPI lifespan starts MCP, `max_containers=1`, `@modal.concurrent`), rate limit.
- Files: `gspaenli/ui.py`, `gspaenli/ratelimit.py`, `modal_app.py`, `tests/test_ratelimit.py`.
- Done when: `modal deploy modal_app.py` gives a public URL; the definition-of-done question works in EN and FR in each persona; the trace shows tool, args, source, latency, model.

### PR 5: README

- Scope: live link, what it is, architecture diagram, quickstart (local + Modal), switching models, "Why prompting, not finetuning" paragraph, known limits, next steps (multi-step tools, voice, static front end).
- Done when: a stranger can read it in 2 minutes; style check: no em-dashes, no fluff.

### PR 6 (optional): evals Apertus vs Qwen3-8B

- Scope: `evals/run.py` measures tool accuracy, reply-language accuracy, latency; results table in README. Redeploy LLM with `MODEL_ID=Qwen/Qwen3-8B` once, run, switch back.
- Done when: table committed; GPU stopped.

### PR 7 (optional, next step): voice and static front end

- ElevenLabs: `/speak` (TTS of the answer, multilingual voice so FR/DE/IT answers sound right) and `/transcribe` (Scribe STT), key in Modal secret `elevenlabs`, loaded lazily so text-only mode works without it.
- Static GitHub Pages front end calling a JSON `/api/chat` endpoint on the same Modal app (CORS restricted to the Pages origin). Gradio stays as the fallback.
- Done when: voice works in the browser for EN and FR; Pages demo loads without a Modal cold start on the page itself.

### Time budget

PR1 45 min, PR2 40 min, PR3 75 min, PR4 50 min, PR5 20 min. Optional items only if time is left.

## Changes after approval (2026-10-13)

Agreed during implementation:

- **PR 5: UI polish and guided tour** (new). Persona cards, Swiss visual identity (light and dark),
  persona avatars, empty state, model status pill, mobile layout, and a step-by-step in-app tour with
  Skip, shown once per browser. The status pill never polls the GPU: any request to the LLM endpoint,
  even `/health`, wakes it and costs credit.
- **PR 6: README and landing page.** The README as planned, plus a static landing page on GitHub
  Pages (`docs/`) that links to the Modal app. The app stays on Modal because it needs secrets, the
  MCP subprocess and the GPU.
- **PR 7 (optional): evals**, Apertus vs an alternative model (was PR 6).
- **PR 8 (optional): voice and a static chat front end** (was PR 7).
