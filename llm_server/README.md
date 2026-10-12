# LLM server

vLLM on Modal with an OpenAI-compatible API, protected by a bearer token. Scales to zero after 5 idle minutes, at most one GPU container.

## Deploy

Needs the Modal secret `gspaenli-llm` with `LLM_API_KEY`.

```bash
uv run modal deploy llm_server/serve.py
```

| Env var at deploy | Default | Notes |
|---|---|---|
| `MODEL_ID` | `swiss-ai/Apertus-8B-Instruct-2509` | any vLLM-supported Hugging Face model |
| `GPU` | `L4,A10` | comma-separated fallback list, tried in order; A10 is faster and 1.4x the price |
| `MAX_MODEL_LEN` | `8192` | lower it if a model does not fit |
| `HF_SECRET` | empty | name of a Modal secret with `HF_TOKEN`, only for gated models |

The URL is printed by `modal deploy`, for example `https://<workspace>--gspaenli-llm-serve.modal.run`.

## Test

```bash
curl -s $URL/health -o /dev/null -w "%{http_code}\n"
curl -s $URL/v1/models -H "Authorization: Bearer $LLM_API_KEY"
curl -s $URL/v1/chat/completions -H "Authorization: Bearer $LLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"swiss-ai/Apertus-8B-Instruct-2509","messages":[{"role":"user","content":"Bonjour, qui es-tu ?"}],"max_tokens":80}'
```

The first request after idle waits for a cold start (weights load from the Volume). Without the token, `/v1/*` returns 401.

## Stop

Scale to zero is automatic. To remove the endpoint completely: `uv run modal app stop gspaenli-llm`.
