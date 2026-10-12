"""Starts the real Swiss MCP over stdio. Run with: uv run pytest -m integration"""

import json

import pytest

from gspaenli.config import Settings
from gspaenli.mcp_client import ALLOWED_TOOLS, SwissMCP

pytestmark = pytest.mark.integration


async def test_lists_allowed_tools_and_calls_offline_tools():
    async with SwissMCP(Settings.from_env().mcp_dir) as mcp:
        assert set(mcp.tools) <= ALLOWED_TOOLS
        assert "swiss_health_insurance_premiums" in mcp.tools
        assert "crawl_federal_sources" not in mcp.tools

        res = await mcp.call(
            "swiss_health_insurance_premiums", {"age": 30, "municipality_code": 5586, "franchise": 300}
        )
        data = json.loads(res.text)
        assert not res.is_error
        assert data["municipality"] == "Lausanne"
        assert res.sources

        res = await mcp.call("search_knowledge", {"query": "waste collection", "limit": 2})
        assert not res.is_error and res.text
