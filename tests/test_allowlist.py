import json

import pytest

from gspaenli.config import ROOT
from gspaenli.mcp_client import (
    ALLOWED_TOOLS,
    SwissMCP,
    ToolInfo,
    ToolNotAllowedError,
    extract_sources,
    filter_allowed,
    truncate,
)
from gspaenli.personas import load_personas

EXCLUDED = {
    "crawl_federal_sources",
    "crawl_cantonal_sources",
    "crawl_municipal_sources",
    "open_dashboard",
    "source_status",
}


def test_excluded_tools_are_not_allowed():
    assert not EXCLUDED & ALLOWED_TOOLS


def test_filter_keeps_only_allowed_tools():
    tools = [ToolInfo(n, "", {}) for n in ["search_knowledge", "crawl_federal_sources", "unknown_new_tool"]]
    assert [t.name for t in filter_allowed(tools)] == ["search_knowledge"]


def test_persona_preferred_tools_are_allowed():
    for persona in load_personas(ROOT / "personas").values():
        assert set(persona.preferred_tools) <= ALLOWED_TOOLS, persona.id


async def test_call_refuses_tool_outside_allow_list(tmp_path):
    mcp = SwissMCP(tmp_path)
    with pytest.raises(ToolNotAllowedError, match="allow-list"):
        await mcp.call("crawl_federal_sources", {})


async def test_call_refuses_allowed_but_unregistered_tool(tmp_path):
    mcp = SwissMCP(tmp_path)
    mcp.tools = {"search_knowledge": ToolInfo("search_knowledge", "", {})}
    with pytest.raises(ToolNotAllowedError, match="not available"):
        await mcp.call("compute_swiss_tax", {"kanton_shortcode": "ZH"})


def test_missing_tools_reports_unregistered(tmp_path):
    mcp = SwissMCP(tmp_path)
    mcp.tools = {"search_knowledge": ToolInfo("search_knowledge", "", {})}
    assert mcp.missing_tools(["search_knowledge", "compute_swiss_tax"]) == ["compute_swiss_tax"]


def test_truncate():
    assert truncate("abc", 5) == ("abc", False)
    text, cut = truncate("x" * 10, 5)
    assert cut and text.startswith("xxxxx") and "truncated" in text


def test_extract_sources_from_nested_json():
    payload = {
        "premiums": [{"source_url": "https://a.example/1"}, {"source_url": "https://a.example/1"}],
        "official_reference": "https://b.example",
        "results": [{"url": "https://c.example", "passage": "not a url key"}],
        "source": "swisstopo",
    }
    assert extract_sources(json.dumps(payload)) == [
        "https://a.example/1",
        "https://b.example",
        "https://c.example",
    ]


def test_extract_sources_ignores_non_json():
    assert extract_sources("plain text with https://x.example") == []
