import json
from typing import Any

import pytest

from gspaenli.config import ROOT
from gspaenli.mcp_client import ToolInfo, ToolResult, extract_sources
from gspaenli.personas import load_personas


class FakeLLM:
    """Returns scripted replies in order and records every request."""

    model_id = "fake/model"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, *, max_tokens, temperature, json_schema=None) -> str:
        self.calls.append({"messages": messages, "json_schema": json_schema})
        return self.replies.pop(0)


class FakeMCP:
    def __init__(self, results: dict[str, Any] | None = None) -> None:
        names = [
            "swiss_health_insurance_premiums",
            "search_knowledge",
            "swiss_school_holidays",
            "swiss_geo_search",
            "swiss_geo_context",
            "swiss_federal_political_rights",
        ]
        self.tools = {
            n: ToolInfo(n, f"{n} description", {"type": "object", "properties": {}, "required": []})
            for n in names
        }
        self.tools["swiss_school_holidays"].input_schema = {
            "type": "object",
            "properties": {"canton": {"type": "string"}, "school_year": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
            "required": ["canton"],
        }
        self.tools["search_knowledge"].input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        }
        self.results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        payload = self.results.get(name, {"status": "answered", "source_url": f"https://example.ch/{name}"})
        if isinstance(payload, Exception):
            raise payload
        text = json.dumps(payload)
        return ToolResult(name=name, arguments=arguments, text=text, sources=extract_sources(text))


@pytest.fixture
def personas():
    return load_personas(ROOT / "personas")


@pytest.fixture
def places_db(tmp_path):
    """Tiny copy of the MCP places schema, so resolver tests do not need vendor/."""
    import sqlite3

    data = tmp_path / "data"
    data.mkdir()
    conn = sqlite3.connect(data / "swiss_places_premiums_2026.sqlite")
    conn.executescript(
        """
        CREATE TABLE communes (bfs_code INTEGER PRIMARY KEY, name TEXT, normalized_name TEXT, canton TEXT);
        CREATE TABLE place_aliases (alias TEXT, normalized_alias TEXT, bfs_code INTEGER, alias_type TEXT);
        INSERT INTO communes VALUES (5586, 'Lausanne', 'lausanne', 'VD'), (6621, 'Genève', 'geneve', 'GE'),
                                    (3203, 'St. Gallen', 'st gallen', 'SG');
        INSERT INTO place_aliases VALUES ('Lausanne', 'lausanne', 5586, 'current'),
                                         ('Genève', 'geneve', 6621, 'current'),
                                         ('St. Gallen', 'st gallen', 3203, 'current');
        """
    )
    conn.commit()
    conn.close()
    return tmp_path
