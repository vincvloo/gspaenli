"""Public Gradio app on Modal (CPU). Talks to the LLM server and runs the Swiss MCP in-container.

    uv run modal deploy modal_app.py

Needs the Modal secret "gspaenli-app" with LLM_BASE_URL, LLM_API_KEY, MODEL_ID
(scripts/create_app_secret.py creates it from your .env).
"""

import modal

MCP_REPO = "https://github.com/dantevier/swiss-ai-week-mcp.git"
MCP_COMMIT = "be988a0e02d9f66163f4f9bea6f786be3209a599"  # tag v1.0
MCP_DIR = "/opt/swiss-ai-week-mcp"
MINUTES = 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands("pip install uv==0.12.18")  # the MCP runs as `uv run --frozen ...`
    .run_commands(
        f"git clone --branch v1.0 --depth 1 {MCP_REPO} {MCP_DIR}",
        f'test "$(git -C {MCP_DIR} rev-parse HEAD)" = "{MCP_COMMIT}"',
        f"uv sync --frozen --no-dev --directory {MCP_DIR}",
        # Build the MCP's local SQLite data now, so the first request does not wait ~25 s.
        f"cd {MCP_DIR} && uv run --frozen python -c "
        '"from mcp_swiss_info.local_databases import build_local_databases as b; b()"',
    )
    .uv_sync()  # app dependencies from uv.lock
    .env(
        {
            "MCP_DIR": MCP_DIR,
            "PERSONAS_DIR": "/root/personas",
            "GRADIO_ANALYTICS_ENABLED": "False",
        }
    )
    .add_local_dir("personas", "/root/personas")
    .add_local_python_source("gspaenli")
)

app = modal.App("gspaenli")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gspaenli-app")],
    max_containers=1,  # Gradio needs sticky sessions; one CPU container is plenty for a demo
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    cpu=1.0,
    memory=1024,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def ui():
    from gspaenli.ui import create_app

    return create_app()
