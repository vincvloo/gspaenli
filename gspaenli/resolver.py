"""Turn model-friendly arguments into the arguments the MCP tools expect.

The model cannot know that Lausanne is BFS 5586, so for the premium tool it gives a municipality
name and we look the code up in the MCP's own places database (read-only). Unknown names fall
back to swiss_geo_search + swiss_geo_context. Defaults are stated as notes in the answer.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .mcp_client import ToolInfo, ToolResult

PREMIUM_TOOL = "swiss_health_insurance_premiums"
# Name filters that only match some languages ("autumn" works, "Herbstferien" does not).
# Without them the result is a short list for the whole school year, and the answer step picks.
DROP_ARGUMENTS = {"swiss_school_holidays": {"holiday_type"}}
PLACES_DB = Path("data") / "swiss_places_premiums_2026.sqlite"
DEFAULT_AGE = 30
DEFAULT_FRANCHISE = 300
FRANCHISES = (300, 500, 1000, 1500, 2000, 2500)

# Names in other languages for places the database only knows in the local language.
EXONYMS = {
    "geneva": "geneve", "genf": "geneve", "ginevra": "geneve",
    "losanna": "lausanne",
    "zurigo": "zurich",
    "berne": "bern", "berna": "bern",
    "bale": "basel", "basilea": "basel", "basle": "basel",
    "lucerne": "luzern", "lucerna": "luzern",
    "friburgo": "fribourg", "freiburg": "fribourg",
    "neuenburg": "neuchatel",
    "sitten": "sion",
    "sankt gallen": "st gallen", "san gallo": "st gallen", "saint gall": "st gallen",
    "biel": "biel bienne", "bienne": "biel bienne",
    "coira": "chur", "coire": "chur",
    "lauis": "lugano",
    "bellenz": "bellinzona",
}

CANTONS = {
    "AG": ["aargau", "argovie", "argovia"], "AI": ["appenzell innerrhoden", "appenzell rhodes interieures"],
    "AR": ["appenzell ausserrhoden", "appenzell rhodes exterieures"], "BE": ["bern", "berne", "berna"],
    "BL": ["basel landschaft", "bale campagne", "basilea campagna"], "BS": ["basel stadt", "bale ville", "basilea citta"],
    "FR": ["fribourg", "freiburg", "friburgo"], "GE": ["geneve", "geneva", "genf", "ginevra"],
    "GL": ["glarus", "glaris", "glarona"], "GR": ["graubunden", "grisons", "grigioni"],
    "JU": ["jura", "giura"], "LU": ["luzern", "lucerne", "lucerna"], "NE": ["neuchatel", "neuenburg"],
    "NW": ["nidwalden", "nidwald", "nidvaldo"], "OW": ["obwalden", "obwald", "obvaldo"],
    "SG": ["st gallen", "sankt gallen", "saint gall", "san gallo"], "SH": ["schaffhausen", "schaffhouse", "sciaffusa"],
    "SO": ["solothurn", "soleure", "soletta"], "SZ": ["schwyz", "svitto"], "TG": ["thurgau", "thurgovie", "turgovia"],
    "TI": ["ticino", "tessin"], "UR": ["uri"], "VD": ["vaud", "waadt"], "VS": ["valais", "wallis", "vallese"],
    "ZG": ["zug", "zoug", "zugo"], "ZH": ["zurich", "zuerich", "zurigo"],
}
CANTON_BY_NAME = {name: code for code, names in CANTONS.items() for name in names}


class ToolCaller(Protocol):
    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


@dataclass
class Resolution:
    arguments: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    extra_calls: list[str] = field(default_factory=list)
    error: str | None = None


def normalize(name: str) -> str:
    """Same scheme as the MCP database: no accents, lowercase, punctuation to spaces."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def canton_code(value: str) -> str | None:
    raw = value.strip()
    if raw.upper().removeprefix("CH-") in CANTONS:
        return raw.upper().removeprefix("CH-")
    return CANTON_BY_NAME.get(normalize(raw).removeprefix("canton of ").removeprefix("canton de ").removeprefix("kanton "))


def router_tools(tools: dict[str, ToolInfo]) -> dict[str, ToolInfo]:
    """Tool list as shown to the router: the premium tool takes a municipality name, not a code."""
    shown = dict(tools)
    if PREMIUM_TOOL in tools:
        shown[PREMIUM_TOOL] = ToolInfo(
            name=PREMIUM_TOOL,
            description=(
                "2026 minimum basic health insurance premiums (KVG/LAMal) for a Swiss municipality. "
                "municipality: place name as the user wrote it. age: only if the user gave it. "
                f"franchise: annual deductible in CHF {FRANCHISES}, only if the user gave it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "municipality": {"type": "string"},
                    "age": {"type": "integer"},
                    "franchise": {"type": "integer"},
                },
                "required": ["municipality"],
            },
        )
    return shown


def lookup_bfs(name: str, mcp_dir: Path) -> tuple[int, str] | None:
    """(bfs_code, official name) for a municipality name, or None."""
    db = Path(mcp_dir) / PLACES_DB
    if not db.is_file():
        return None
    key = normalize(name)
    key = EXONYMS.get(key, key)
    with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT a.bfs_code, c.name FROM place_aliases a JOIN communes c ON c.bfs_code = a.bfs_code "
            "WHERE a.normalized_alias = ? ORDER BY a.alias_type = 'current' DESC LIMIT 1",
            (key,),
        ).fetchone()
    return (int(row[0]), str(row[1])) if row else None


async def geo_lookup_bfs(name: str, mcp: ToolCaller, calls: list[str]) -> tuple[int, str] | None:
    """Fallback through the MCP: place search for coordinates, then municipality at that point."""
    found = await mcp.call("swiss_geo_search", {"query": name, "limit": 1})
    calls.append(f"swiss_geo_search({name!r})")
    try:
        loc = json.loads(found.text)["locations"][0]
        ctx = await mcp.call("swiss_geo_context", {"latitude": loc["latitude"], "longitude": loc["longitude"]})
        calls.append("swiss_geo_context")
        muni = json.loads(ctx.text)["municipality"]
        return int(muni["bfs_number"]), str(muni["name"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None


def coerce_arguments(arguments: dict[str, Any], tool: ToolInfo) -> dict[str, Any]:
    """Drop unknown keys and cast simple types; small models often send "30" for 30."""
    props = tool.input_schema.get("properties", {})
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if key not in props or value is None:
            continue
        types = {props[key].get("type")} | {t.get("type") for t in props[key].get("anyOf", [])}
        try:
            if "integer" in types and isinstance(value, (str, float)):
                value = int(float(value))
            elif "number" in types and isinstance(value, str):
                value = float(value)
            elif "string" in types and isinstance(value, (int, float)):
                value = str(value)
        except ValueError:
            pass
        out[key] = value
    return out


async def resolve(
    tool_name: str, arguments: dict[str, Any], tools: dict[str, ToolInfo], mcp: ToolCaller, mcp_dir: Path
) -> Resolution:
    if tool_name == PREMIUM_TOOL:
        return await _resolve_premium(arguments, mcp, mcp_dir)
    args = {k: v for k, v in arguments.items() if k not in DROP_ARGUMENTS.get(tool_name, set())}
    if isinstance(args.get("canton"), str) and (code := canton_code(args["canton"])):
        args["canton"] = code
    tool = tools.get(tool_name)
    return Resolution(arguments=coerce_arguments(args, tool) if tool else args)


async def _resolve_premium(arguments: dict[str, Any], mcp: ToolCaller, mcp_dir: Path) -> Resolution:
    notes: list[str] = []
    calls: list[str] = []
    name = arguments.get("municipality") or arguments.get("municipality_name")
    code = arguments.get("municipality_code")
    if isinstance(code, str) and not code.strip().isdigit():
        name, code = name or code, None

    if code is not None and not name:
        bfs = int(code)
    elif name:
        found = lookup_bfs(str(name), mcp_dir) or await geo_lookup_bfs(str(name), mcp, calls)
        if not found:
            return Resolution(arguments, extra_calls=calls, error=f"Unknown Swiss municipality: {name}")
        bfs, official = found
        notes.append(f"{name} is municipality {official} (BFS {bfs})")
    else:
        return Resolution(arguments, error="No municipality given")

    try:
        age = int(arguments["age"])
    except (KeyError, TypeError, ValueError):
        age = DEFAULT_AGE
        notes.append(f"Assumed an adult aged {DEFAULT_AGE} (adult rate from age 26)")
    try:
        franchise = int(arguments["franchise"])
        if franchise not in FRANCHISES:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        franchise = DEFAULT_FRANCHISE
        notes.append(
            f"Assumed the {DEFAULT_FRANCHISE} CHF franchise (deductible), the standard one; "
            "a higher franchise lowers the premium"
        )
    return Resolution({"age": age, "municipality_code": bfs, "franchise": franchise}, notes, calls)
