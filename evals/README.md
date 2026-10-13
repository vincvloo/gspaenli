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
| Apertus 8B Instruct (`swiss-ai/Apertus-8B-Instruct-2509`) | 12/12 | 12/12 | 8/12 | 2 | 6.7 s |
| Qwen3 8B (`Qwen/Qwen3-8B`, thinking off) | 12/12 | 12/12 | 11/12 | 0 | 10.7 s |

Both on a single L4 GPU with vLLM 0.21.0, warm. Raw results with answers and tool output: [results/](results/). The first run, before the fixes below, is in [results/baseline/](results/baseline/).

### Per question

| Question | Apertus | Qwen | Notes |
|---|---|---|---|
| premium_en | ✓ | ✓ | |
| premium_fr | ✓ | ✓ | |
| premium_de_young | ✓ | ✓ | Apertus quoted the with-accident price as cheapest before the summary line |
| premium_it | ✓ | ✓ | |
| permit_fr | ~ | ✓ | Apertus adds generic steps not in the source |
| initiative_en | ✓ | ✓ | |
| national_council_it | ~ | ✓ | Apertus names the wrong office as source |
| parcel_vat_de | ✓ | ✓ | |
| reference_rate_en | ✓ | ✓ | |
| holidays_fr | ✓ | ✓ | |
| licence_en | ✗ | ✓ | Apertus followed "explain Swiss terms" into an invented glossary |
| waste_de | ✗ | ✗ | the data has no waste schedule for Bern; both models improvise |

✓ every fact is in the tool result. ~ correct core answer with an unsupported detail. ✗ an invented or wrong fact.

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

## Reading

- **Routing and language are solved** for this question set, mostly by code around the model: argument repair, a stopword language check, and JSON schema output.
- **Grounding is the real gap.** Qwen3 8B stays closer to the data; Apertus 8B fills gaps from memory more often, especially when the tool result is thin (permits, waste).
- **Apertus takes about 40% less time per answer** here (6.7 s vs 10.7 s), partly because its answers are shorter.
- The demo keeps **Apertus as the default**: it is the Swiss open model this project is about, and it is faster. Switching is one deploy parameter (`MODEL_ID=Qwen/Qwen3-8B`).
- 12 questions is a smoke test, not a benchmark. One manual grader (Claude), one run per model.
