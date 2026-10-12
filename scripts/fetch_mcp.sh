#!/usr/bin/env sh
# Clone the Swiss MCP at the pinned commit (tag v1.0) into vendor/ and install its deps.
# The Modal image does the same at build time.
set -eu

MCP_REPO="https://github.com/dantevier/swiss-ai-week-mcp.git"
MCP_COMMIT="be988a0e02d9f66163f4f9bea6f786be3209a599"
DEST="${1:-vendor/swiss-ai-week-mcp}"

if [ ! -d "$DEST/.git" ]; then
  git clone --branch v1.0 --depth 1 "$MCP_REPO" "$DEST"
fi
actual="$(git -C "$DEST" rev-parse HEAD)"
if [ "$actual" != "$MCP_COMMIT" ]; then
  echo "MCP is at $actual, expected $MCP_COMMIT" >&2
  exit 1
fi
uv sync --frozen --no-dev --directory "$DEST"
echo "Swiss MCP ready in $DEST"
