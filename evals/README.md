# Evals

12 questions in [questions.yaml](questions.yaml): English, French, German and Italian, two where the source is in another language than the question. Each one runs the real pipeline (route, resolve, one MCP call, answer in persona) against a model on Modal.

```bash
uv run python evals/run.py                                      # model from .env
uv run python evals/run.py --model Qwen/Qwen3-8B --base-url <url>/v1
```

`run.py` scores two things automatically: did the router pick the expected tool, and is the answer in the language of the question. A third score is manual: **is every fact in the answer supported by the tool result?** The automatic scores alone were misleading: the first Apertus run picked the right tool 11 times out of 12, but several answers came from the model's memory.

## Results (2026-10-13, final code)

| Model | Right tool | Right language | Fully grounded | Partly | Median time |
|---|---|---|---|---|---|
| Apertus 8B Instruct (`swiss-ai/Apertus-8B-Instruct-2509`) | 12/12 | 12/12 | 10/12 | 2 | 10.4 s |
| Qwen3 8B (`Qwen/Qwen3-8B`, thinking off) | 12/12 | 12/12 | 11/12 | 1 | 11.6 s |

Both on a single L4 GPU with vLLM 0.21.0, warm. Median time varies between runs (Apertus: 6.2 to 13.6 s over four runs), so treat it as rough. Raw results with answers and tool output: [results/](results/). The first run, before any fixes, is in [results/baseline/](results/baseline/).

### Progress

| Run | Apertus grounded | Qwen grounded |
|---|---|---|
| First run, no guards | about 6/12 | about 8/12 |
| After argument repair and missing-data handling | 8/12 | 11/12 |
| After relevance check, no-data answers and persona fix (final) | 10/12 | 11/12 |

### Per question (final run)

| Question | Apertus | Qwen | Notes |
|---|---|---|---|
| premium_en | ✓ | ✓ | |
| premium_fr | ✓ | ✓ | |
| premium_de_young | ~ | ✓ | Apertus calls the monthly premium "jährlich" (yearly) |
| premium_it | ✓ | ✓ | |
| permit_fr | ✓ | ✓ | |
| initiative_en | ✓ | ✓ | |
| national_council_it | ✓ | ~ | Qwen says "Consiglio dei Cantoni" for Consiglio degli Stati |
| parcel_vat_de | ✓ | ✓ | |
| reference_rate_en | ~ | ✓ | Apertus adds a "rent index" that is not in the data |
| holidays_fr | ✓ | ✓ | |
| licence_en | ✓ | ✓ | the invented glossary is gone after the persona fix |
| waste_de | ✓ | ✓ | no data for Bern: both now say so instead of inventing a schedule |

✓ every fact is in the tool result. ~ correct core answer with an unsupported or wrong detail. ✗ an invented or wrong fact.

## What the evals changed in the code

Every fix is generic and has a unit test in [tests/test_guards.py](../tests/test_guards.py).

| Found | Fix |
|---|---|
| The MCP returns `status: invalid_input` as a normal result; Apertus then answered from memory | Any non-`answered` status counts as missing data, and the answer prompt says not to answer from memory |
| "topic must be one of: initiative, ...", the model sent "federal popular initiative" | Map the value to the closest allowed option and retry once |
| Validation errors like `fact_type` 'fees' (allowed: 'fee', ...) or `language` 'en' (allowed: it, de, fr, rm) | Same mapping; an optional argument with no sensible match is dropped |
| `vat_rate: "8.1%"` rejected, error names the argument | Drop optional arguments named in the error and retry |
| Qwen invented `as_of: 2023-10-10` for "the current rate" | Drop dates the user did not write |
| Router sent `municipality: "None"` as a string | Drop "None" and "null" strings |
| Apertus invented coordinates for Bern to call `swiss_geo_context` | No coordinates in the question: use full-text search instead |
| Apertus quoted the wrong premium as cheapest | A one-line summary on top of the premium result |
| Qwen treated someone from Germany as non-EU | The permit tool's description, as the router sees it, names EU/EFTA countries |
| Answers ending with "(TOOL RESULT)" | Removed from answers |
| Keyword search matched Bern's moving page on the word "Bern" alone; both models invented a waste schedule | Relevance check: a topic word of the question (not a place name, not a stopword) must appear in the passages, otherwise it counts as no data |
| With an error or irrelevant result in the prompt, models still answered from memory | A separate no-data prompt with no tool output and a 150-token limit: say it is not in the official data, point to the commune, give no facts |
| Qwen chose no tool for the waste question and invented a URL | A message with three or more topic words is a question: search the saved pages instead of answering without a tool |
| The Newcomer prompt listed example terms; Apertus turned them into an invented glossary | Explain only terms the answer actually uses, never a list |
| The eval counted an Italian answer as French ("La", "Le") | More Italian and French marker words in the language check (also used by the app) |

## Reading

- **Routing and language are solved** for this question set, mostly by code around the model: argument repair, a stopword language check, JSON schema output.
- **Grounding improved most from code, not prompts**: checking that a lookup is relevant, and answering "not found" with a prompt that has nothing to improvise from.
- **Qwen3 8B still stays a bit closer to the data**; Apertus 8B adds small unsupported details more often.
- The demo keeps **Apertus as the default**: it is the Swiss open model this project is about, and the gap is now small. Switching is one deploy parameter (`MODEL_ID=Qwen/Qwen3-8B`).
- **Trade-off of the relevance check:** a question in one language against saved pages in another can now get an honest "not found" instead of an answer. That is a safe failure, but a failure.
- 12 questions is a smoke test, not a benchmark. One manual grader (Claude), one run per model and code version.
