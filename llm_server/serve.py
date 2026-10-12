"""OpenAI-compatible vLLM server on Modal, protected by a bearer token.

Deploy (defaults: Apertus 8B on an L4):
    uv run modal deploy llm_server/serve.py
Other model or GPU, no code change:
    MODEL_ID=Qwen/Qwen3-8B GPU=A10 uv run modal deploy llm_server/serve.py

Deploy-time settings are baked into the image env so the container sees the same values.
"""

import os
import subprocess

import modal

MODEL_ID = os.environ.get("MODEL_ID", "swiss-ai/Apertus-8B-Instruct-2509")
# Comma-separated, tried in order. L4 is the cheapest GPU that fits 8B in bf16 (24 GB);
# A10 (also 24 GB) is the fallback when Modal has no L4 capacity.
GPU = os.environ.get("GPU", "L4,A10")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "8192")
# Optional Hugging Face token secret, only needed for gated models or rate limits.
HF_SECRET = os.environ.get("HF_SECRET", "")

VLLM_PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "MODEL_ID": MODEL_ID,
            "GPU": GPU,
            "MAX_MODEL_LEN": MAX_MODEL_LEN,
            "HF_SECRET": HF_SECRET,
        }
    )
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

secrets = [modal.Secret.from_name("gspaenli-llm")]  # LLM_API_KEY
if HF_SECRET:
    secrets.append(modal.Secret.from_name(HF_SECRET))  # HF_TOKEN

app = modal.App("gspaenli-llm")


@app.function(
    image=image,
    gpu=[g.strip() for g in GPU.split(",")],
    secrets=secrets,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    max_containers=1,  # hard cap on GPU spend
    scaledown_window=5 * MINUTES,  # scale to zero after 5 idle minutes
    timeout=10 * MINUTES,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    # vLLM reads VLLM_API_KEY and then requires "Authorization: Bearer <key>" on /v1/*.
    # /health stays open, which lets the app check whether the model is awake.
    env = {**os.environ, "VLLM_API_KEY": os.environ["LLM_API_KEY"]}
    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--served-model-name",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        MAX_MODEL_LEN,
        "--gpu-memory-utilization",
        "0.92",
        "--enforce-eager",  # skip CUDA graph capture: faster cold start, slightly slower decode
        "--uvicorn-log-level",
        "warning",
    ]
    subprocess.Popen(cmd, env=env)
