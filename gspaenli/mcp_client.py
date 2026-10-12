"""Client for the Swiss MCP, run as a stdio subprocess, restricted to an allow-list of tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)

# Allow-list, not deny-list: crawlers, dashboard and source_status stay out of the agent.
ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "swiss_health_insurance_premiums",
        "swiss_residence_permit_guidance",
        "swiss_housing_info",
        "swiss_reference_interest_rate",
        "get_driving_licence_exchange_info",
        "swiss_school_holidays",
        "compute_swiss_tax",  # not registered at MCP v1.0; used if a later tag adds it
        "company_info",
        "swiss_geo_search",
        "swiss_geo_context",
        "swiss_weather_forecast",
        "swiss_weather_observations",
        "swiss_federal_political_rights",
        "swiss_import_parcel_vat",
        "bfs_population",
        "search_knowledge",
        "get_source",
        "opendata_search_datasets",
        "opendata_dataset",
    }
)

SOURCE_KEYS = ("source_url", "url", "official_reference", "map_url")
DEFAULT_MAX_CHARS = 3000


class ToolNotAllowedError(ValueError):
    pass


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    name: str
    arguments: dict[str, Any]
    text: str
    is_error: bool = False
    truncated: bool = False
    sources: list[str] = field(default_factory=list)
    latency_s: float = 0.0


def filter_allowed(tools: list[ToolInfo]) -> list[ToolInfo]:
    return [t for t in tools if t.name in ALLOWED_TOOLS]


def truncate(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n[... truncated]", True


def extract_sources(text: str, limit: int = 5) -> list[str]:
    """Collect source URLs from a JSON tool result so citations come from data, not the model."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in SOURCE_KEYS and isinstance(value, str) and value.startswith("http"):
                    if value not in found:
                        found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return found[:limit]


def server_params(mcp_dir: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command="uv",
        args=["run", "--frozen", "--directory", str(mcp_dir), "python", "-m", "mcp_swiss_info.main"],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


class SwissMCP:
    """One long-lived stdio session. Calls are serialized; the POC does one tool call per turn."""

    def __init__(self, mcp_dir: Path, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self.mcp_dir = Path(mcp_dir)
        self.max_chars = max_chars
        self.tools: dict[str, ToolInfo] = {}
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if not (self.mcp_dir / "pyproject.toml").is_file():
            raise FileNotFoundError(f"Swiss MCP not found in {self.mcp_dir}; run scripts/fetch_mcp.sh")
        self._stack = AsyncExitStack()
        try:
            errlog = self._stack.enter_context(open(os.devnull, "w"))  # noqa: SIM115
            if os.environ.get("GSPAENLI_MCP_LOG"):
                errlog = sys.stderr
            read, write = await self._stack.enter_async_context(
                stdio_client(server_params(self.mcp_dir), errlog=errlog)
            )
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
            listed = (await self._session.list_tools()).tools
        except BaseException:
            await self.close()
            raise
        all_tools = [ToolInfo(t.name, t.description or "", t.input_schema) for t in listed]
        self.tools = {t.name: t for t in filter_allowed(all_tools)}
        log.info("Swiss MCP ready: %d of %d tools allowed", len(self.tools), len(all_tools))

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()
        self._stack = self._session = None

    async def __aenter__(self) -> SwissMCP:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def missing_tools(self, names: list[str]) -> list[str]:
        """Names that are not available (not allowed or not registered on this MCP version)."""
        return [n for n in names if n not in self.tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in ALLOWED_TOOLS:
            raise ToolNotAllowedError(f"tool {name!r} is not on the allow-list")
        if name not in self.tools:
            raise ToolNotAllowedError(f"tool {name!r} is not available on this MCP server")
        if self._session is None:
            raise RuntimeError("SwissMCP.start() was not called")
        started = time.perf_counter()
        async with self._lock:
            result = await self._session.call_tool(name, arguments)
        text = "".join(getattr(c, "text", "") for c in result.content)
        short, truncated = truncate(text, self.max_chars)
        return ToolResult(
            name=name,
            arguments=arguments,
            text=short,
            is_error=bool(result.is_error),
            truncated=truncated,
            sources=extract_sources(text),
            latency_s=round(time.perf_counter() - started, 2),
        )


async def _demo() -> None:
    from .config import Settings

    settings = Settings.from_env()
    async with SwissMCP(settings.mcp_dir) as mcp:
        for name in sorted(mcp.tools):
            print(name)
        res = await mcp.call(
            "swiss_health_insurance_premiums", {"age": 30, "municipality_code": 5586, "franchise": 300}
        )
        print(f"\nLausanne, adult, franchise 300 ({res.latency_s}s):\n{res.text}\nsources: {res.sources}")


if __name__ == "__main__":
    asyncio.run(_demo())
