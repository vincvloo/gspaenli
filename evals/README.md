# Evals

16 questions in [questions.yaml](questions.yaml): English, French, German and Italian, two where the source is in another language than the question, and four **multi-step** questions that need two lookups (one per language). Each one runs the real pipeline (route, one to three MCP lookups, answer in persona) against a model on Modal.

```bash
uv run python evals/run.py                                      # model from .env
uv run python evals/run.py --model Qwen/Qwen3-8B --base-url <url>/v1
uv run python evals/rescore.py                                  # recompute automatic scores of saved results
```

`run.py` scores automatically: did the lookups include the expected tools (for multi-step questions, all of `required_tools`), how many lookups were made, and is the answer in the language of the question. A third score is manual: **is every fact in the answer supported by the tool results?** The automatic scores alone were misleading: the first Apertus run picked the right tool 11 times out of 12, but several answers came from the model's memory.

## Results (2026-10-13, final code)

| Model | Single-step: right tool | Extra lookups on single-step | Multi-step: all lookups made | Right language | Median time single / multi |
|---|---|---|---|---|---|
| Apertus 8B Instruct (`swiss-ai/Apertus-8B-Instruct-2509`) | 12/12 | 0 | 4/4 | 16/16 | 6.7 s / 10.4 s |
| Qwen3 8B (`Qwen/Qwen3-8B`, thinking off) | 12/12 | 1 | 4/4 | 16/16 | 7.4 s / 10.7 s |

| Model | Fully grounded | Partly | Wrong |
|---|---|---|---|
| Apertus 8B | 12/16 | 3 | 1 |
| Qwen3 8B | 12/16 | 2 | 2 |

Both on a single L4 GPU (A10 as fallback) with vLLM 0.21.0, warm. Times vary a lot between runs on the same code (Apertus single-step: 6.7 to 12.9 s over the last four runs), so treat them as rough. Raw results with answers, lookups and tool output: [results/](results/). The first run, before any fixes and before multi-step, is in [results/baseline/](results/baseline/).

### Progress

| Run | Apertus grounded | Qwen grounded |
|---|---|---|
| First run, 12 questions, no guards | about 6/12 | about 8/12 |
| After argument repair and missing-data handling | 8/12 | 11/12 |
| After relevance check, no-data answers and persona fix | 10/12 | 11/12 |
| Multi-step, 16 questions (final) | 12/16 | 12/16 |

### Per question (final run)

| Question | Apertus | Qwen | Notes |
|---|---|---|---|
| premium_en | ✓ | ✓ | |
| premium_fr | ✓ | ✓ | |
| premium_de_young | ✓ | ✓ | |
| premium_it | ✓ | ✓ | |
| permit_fr | ✗ | ~ | Apertus calls permit B "short-term" (that is permit L); Qwen expands SEM with a wrong name |
| initiative_en | ~ | ✓ | Apertus credits the Federal Office of Justice instead of the Federal Chancellery |
| national_council_it | ✓ | ~ | Qwen says "Consiglio dei Cantoni" for Consiglio degli Stati |
| parcel_vat_de | ✓ | ✓ | |
| reference_rate_en | ✓ | ✓ | |
| holidays_fr | ✓ | ✓ | |
| licence_en | ✓ | ✗ | Qwen invents a theory and a practical test |
| waste_de | ✓ | ✓ | no data for Bern: both say so and point to the commune |
| multi_premiums_compare_en | ✓ | ✓ | Lausanne 531.70 and Geneva 562.00 CHF, both from lookups |
| multi_premium_holidays_fr | ~ | ✗ | both looked up 19 to 23 October; Apertus adds "about two weeks", Qwen writes "23 to 23 October" |
| multi_signatures_de | ~ | ✓ | right figures; Apertus adds that an initiative passes a federal law |
| multi_rate_holidays_it | ✓ | ✓ | rate 1.25% and Ticino holidays 31 October to 8 November |

✓ every fact is in the tool results. ~ correct core answer with an unsupported or wrong detail. ✗ an invented or wrong fact.

## Multi-step lookups

The number of lookups is dynamic: one for most questions, up to three. The Swiss MCP only executes calls; the orchestrator decides. After each lookup it asks the model "is anything missing?" only when the question looks multi-part (a joining word like "and/et/und/e", or several places). That check costs about 3 to 5 s, so single-part questions skip it.

What the multi-step evals found, and the fix for each:

| Found | Fix |
|---|---|
| Apertus said "done" after Lausanne for "compare Lausanne and Geneva", then invented Geneva's premiums | The app detects places in the question that no lookup covered, names them in the "next step?" prompt, and repeats a place-based lookup (premiums, holidays, licence, population) for a missing place itself |
| Apertus wrote `{"missing": "Geneva", "done": false}` but named no tool | Same place expansion; for other gaps, the router runs on the missing part alone, with only the tools not used yet |
| Apertus copied resolved arguments and sent `municipality_code: 6600` for Geneva (it is 6621) | The "next step?" prompt shows arguments as the model wrote them; codes the user did not write are refused |
| "fakultatives Referendum und Volksinitiative" got one lookup; Apertus invented the other figure | A value that names several allowed options gets one lookup per option. German compounds count ("Volksinitiative" names "initiative"), short codes must be whole words ("German" does not name "rm") |
| A gap the model named was never looked up, and Apertus filled it from memory | Gaps never filled become a note to the answer: "could not be checked, give no facts" |
| Apertus repeated lookups with arguments that resolve to the same call | Duplicates are detected after resolution and skipped without an MCP call |
| Qwen used a location lookup as the only result for a waste question and invented rules | Location-only results do not answer a topic question: no-data answer |
| Apertus added a URL that was in no result | Links in answers must appear in the looked-up data; others are removed and listed in the panel |
| "alle" (Italian) was taken for the municipality Alle (JU) | Only capitalised words count as places |
| The "next step?" call cost about 5 s on Apertus for every question | Skipped for single-part questions; replies capped at 100 tokens |
| Apertus called the Geneva premium "missing" after it had been looked up; chasing it added a lookup and a note that Geneva "could not be checked" | A gap whose topic words are already in the results counts as covered |
| For "the rent rate and the Ticino holidays", Apertus said only "Ticino" was missing; the re-route chose housing info and the answer invented holiday dates | An uncovered place gets the place-based tool the question names ("vacanze" means school holidays, "Prämie" premiums, in four languages) |

## What the first evals changed in the code

Every fix is generic and has a unit test in [tests/test_guards.py](../tests/test_guards.py) or [tests/test_multistep.py](../tests/test_multistep.py).

| Found | Fix |
|---|---|
| The MCP returns `status: invalid_input` as a normal result; Apertus then answered from memory | Any non-`answered` status counts as missing data |
| "topic must be one of: initiative, ...", the model sent "federal popular initiative" | Map the value to the closest allowed option and retry once |
| Validation errors like `fact_type` 'fees' (allowed: 'fee', ...) or `language` 'en' (allowed: it, de, fr, rm) | Same mapping; an optional argument with no sensible match is dropped |
| `vat_rate: "8.1%"` rejected, error names the argument | Drop optional arguments named in the error and retry |
| Qwen invented `as_of: 2023-10-10` for "the current rate" | Drop dates the user did not write |
| Router sent `municipality: "None"` as a string | Drop "None" and "null" strings |
| Apertus invented coordinates for Bern to call `swiss_geo_context` | No coordinates in the question: use full-text search instead |
| Apertus quoted the wrong premium as cheapest | A one-line summary on top of the premium result |
| Qwen treated someone from Germany as non-EU | The permit tool's description, as the router sees it, names EU/EFTA countries |
| Answers ending with "(TOOL RESULT)" | Removed from answers |
| Keyword search matched Bern's moving page on the word "Bern" alone; both models invented a waste schedule | Relevance check for text searches (saved pages, housing): a topic word of the question must appear in the passages |
| With an error or irrelevant result in the prompt, models still answered from memory | A separate no-data prompt with no tool output and a 150-token limit |
| Qwen chose no tool for the waste question and invented a URL | A message with three or more topic words is a question: search instead of answering without a tool |
| The Newcomer prompt listed example terms; Apertus turned them into an invented glossary | Explain only terms the answer actually uses |
| The eval counted an Italian answer as French, and a German one as French | More marker words in the language check (also used by the app) |

## Reading

- **Routing, language and multi-step lookups are solved** for this question set, mostly by code around the model: argument repair, place detection, a stopword language check, JSON schema output.
- **Grounding improved most from code, not prompts**: relevance checks, "not found" answers with nothing to improvise from, and only links that come from the data.
- **Answers that combine several results are a new place to slip**: Qwen misread a date range it had looked up correctly.
- **The two models end up level** on grounding (12/16 each) but fail differently: Apertus adds plausible details and wrong office names, Qwen misreads or invents specifics (a date range, a driving test).
- The demo keeps **Apertus as the default**: it is the Swiss open model this project is about. Switching is one deploy parameter (`MODEL_ID=Qwen/Qwen3-8B`).
- **Trade-off of the relevance check:** a question in one language against saved pages in another can get an honest "not found" instead of an answer. A safe failure, but a failure.
- 16 questions is a smoke test, not a benchmark. One manual grader (Claude), one run per model and code version.
