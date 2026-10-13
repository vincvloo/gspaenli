# What a finetune would take

Gspänli uses prompts, not a finetune (see [Why prompting, not finetuning](../README.md#why-prompting-not-finetuning)). This note sketches what a small LoRA finetune of Apertus 8B would need if the project grows, and what it should and should not try to fix. Nothing here has been run; costs are estimates.

## What it should target

The [evals](../evals/README.md) say where the model, not the code, is the limit:

- **Staying inside the tool result.** After the code fixes, Apertus still adds small unsupported details (a "rent index", "yearly" for a monthly premium).
- **Persona voice.** The Bünzli neighbour and the Fedlex clerk sound alike on dry topics.

It should **not** try to teach facts. Premiums, rates and rules change every year; they belong in the Swiss MCP, not in weights. Routing, argument repair and language detection are already handled in code.

## Data

The training example is the answer step: (persona, question, tool result) → answer. Facts come only from the tool result, so the model learns to stay inside it.

| Part | Amount | Source |
|---|---|---|
| Questions | about 150 per persona, in EN, FR, DE, IT | written by hand, then varied by a larger model |
| Tool results | real | run each question through the Swiss MCP with the current router and resolver |
| Answers | one per example, about 450 total | drafted by a larger model, then checked by a person against the tool result |
| No-data cases | about 15% | questions the data does not cover, answered with an honest "not found" |

The human check is the expensive part: about 450 answers at 1 to 2 minutes each, so 8 to 15 hours. Without it, the finetune would learn the larger model's mistakes.

## Training on Modal

- **Method:** LoRA on Apertus 8B (Apache 2.0), rank 16, attention and MLP projections, 2 to 3 epochs, with TRL's SFT trainer and PEFT.
- **Hardware:** one A100 80 GB on Modal ($2.50 per hour at the time of writing). 8B in bf16 with LoRA fits comfortably.
- **Size:** about 450 examples of about 1,000 tokens over 3 epochs is roughly 1.4 million training tokens. That is well under an hour, so a few dollars per run. Plan for 5 to 10 runs while tuning, so tens of dollars in total.
- **Output:** an adapter of a few tens of MB, saved to a Modal Volume.

## Serving

vLLM 0.21.0, already used here, serves LoRA adapters on top of the base model:

```bash
vllm serve swiss-ai/Apertus-8B-Instruct-2509 --enable-lora --max-lora-rank 16 \
  --lora-modules gspaenli-personas=/adapters/gspaenli-personas
```

The app would then use `MODEL_ID=gspaenli-personas`. No other code change: the adapter is just another model name on the same OpenAI-compatible endpoint, and the base model stays available for comparison.

## How to judge it

Run [evals/run.py](../evals/run.py) on the base model and on the adapter, and grade grounding by hand, as in [evals/README.md](../evals/README.md). Extend the set first: 12 questions are enough to find bugs, not to compare two models of similar quality. Worth it only if:

- grounding improves (fewer "partly" and no new "wrong"),
- reply language stays at 100% in all four languages,
- the personas are easier to tell apart in a blind read,
- latency does not get worse.

## Risks

- **Narrower model:** training on 450 short answers can hurt general multilingual quality. Keep the LoRA rank and epochs low and compare on questions outside the training set.
- **Format overfitting:** the model may learn the current tool output formats. When the Swiss MCP changes a format, the adapter may need retraining.
- **Effort is in the data, not the GPU:** the training run costs a few dollars; the checked dataset costs days of work. That is the main reason this project uses prompting.
