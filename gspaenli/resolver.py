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
PERMIT_TOOL = "swiss_residence_permit_guidance"
NULL_STRINGS = {"", "none", "null", "n/a"}
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
    if PERMIT_TOOL in tools:
        base = tools[PERMIT_TOOL]
        shown[PERMIT_TOOL] = ToolInfo(
            name=PERMIT_TOOL,
            description=(
                base.description.strip() + " group: eu_efta for citizens of any EU country or of Iceland, "
                "Liechtenstein or Norway (for example Germany, France, Italy, Austria, Spain, Portugal, "
                "Poland); uk for the United Kingdom; third_country for everyone else (for example USA, India)."
            ),
            input_schema=base.input_schema,
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
        if isinstance(value, str) and value.strip().lower() in NULL_STRINGS:  # router wrote "None"
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


def numbers_in(text: str) -> set[int]:
    # Swiss thousands separators: 2'500 or 2’500
    return {int(n) for n in re.findall(r"\d+", re.sub(r"(?<=\d)['’](?=\d)", "", text))}


async def resolve(
    tool_name: str,
    arguments: dict[str, Any],
    tools: dict[str, ToolInfo],
    mcp: ToolCaller,
    mcp_dir: Path,
    question: str = "",
) -> Resolution:
    if tool_name == PREMIUM_TOOL:
        return await _resolve_premium(arguments, mcp, mcp_dir, numbers_in(question))
    args = {k: v for k, v in arguments.items() if k not in DROP_ARGUMENTS.get(tool_name, set())}
    # Dates the user did not write are guesses ("the current rate" became as_of 2023-10-10).
    invented = [k for k, v in args.items() if isinstance(v, str) and DATE_VALUE.match(v) and v[:4] not in question]
    notes = [f"Dropped {k} {args.pop(k)!r}: no date in the question" for k in invented]
    if isinstance(args.get("canton"), str) and (code := canton_code(args["canton"])):
        args["canton"] = code
    tool = tools.get(tool_name)
    return Resolution(arguments=coerce_arguments(args, tool) if tool else args, notes=notes)


def _user_number(arguments: dict[str, Any], key: str, user_numbers: set[int]) -> int | None:
    """A number the model passed, only if the user actually wrote it. Apertus turned
    "un adulte" into age 18, which has no adult premium."""
    try:
        value = int(arguments[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value in user_numbers else None


async def _resolve_premium(
    arguments: dict[str, Any], mcp: ToolCaller, mcp_dir: Path, user_numbers: set[int]
) -> Resolution:
    notes: list[str] = []
    calls: list[str] = []
    name = arguments.get("municipality") or arguments.get("municipality_name")
    code = arguments.get("municipality_code")
    if isinstance(code, str) and not code.strip().isdigit():
        name, code = name or code, None

    if code is not None and not name:
        if int(code) not in user_numbers:
            # Codes the user did not write are guesses (Apertus sent 6600 for Geneva, it is 6621).
            return Resolution(arguments, error=f"Municipality code {code} is not in the question; a place name is needed")
        bfs = int(code)
    elif name:
        found = lookup_bfs(str(name), mcp_dir) or await geo_lookup_bfs(str(name), mcp, calls)
        if not found:
            return Resolution(arguments, extra_calls=calls, error=f"Unknown Swiss municipality: {name}")
        bfs, official = found
        notes.append(f"{name} is municipality {official} (BFS {bfs})")
    else:
        return Resolution(arguments, error="No municipality given")

    age = _user_number(arguments, "age", user_numbers)
    if age is None:
        age = DEFAULT_AGE
        notes.append(f"Assumed an adult aged {DEFAULT_AGE} (adult rate from age 26)")
    franchise = _user_number(arguments, "franchise", user_numbers)
    if franchise not in FRANCHISES:
        franchise = DEFAULT_FRANCHISE
        notes.append(
            f"Assumed the {DEFAULT_FRANCHISE} CHF franchise (deductible), the standard one; "
            "a higher franchise lowers the premium"
        )
    return Resolution({"age": age, "municipality_code": bfs, "franchise": franchise}, notes, calls)


# --- Guards found by the evals -------------------------------------------------------------

INVALID_CHOICE = re.compile(r"(\w+) must be one of: ([^.]+)")
LITERAL_ERROR = re.compile(r"^(\w+)\n\s+Input should be (.+?) \[type=literal_error", flags=re.M)
DATE_VALUE = re.compile(r"^\d{4}-\d{2}-\d{2}")
OK_STATUSES = {"answered", "ok", "success"}


def result_problem(text: str) -> str | None:
    """A message when a JSON tool result says it did not answer (e.g. status invalid_input).

    The MCP reports these as normal results, so without this the answer step improvised.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or "status" not in data:
        return None
    if str(data["status"]).lower() in OK_STATUSES:
        return None
    return str(data.get("message") or data["status"])


def _invalid_choice(message: str) -> tuple[str, list[str]] | None:
    """(argument, allowed values) from the MCP's own message or a pydantic literal error."""
    if match := INVALID_CHOICE.search(message):
        return match.group(1), [o.strip() for o in match.group(2).split(",") if o.strip()]
    if match := LITERAL_ERROR.search(message):
        return match.group(1), re.findall(r"'([^']+)'", match.group(2))
    return None


def _coverage(value: str, option: str) -> float:
    """Share of the option's words found in the value, as substrings so German compounds count:
    "volksinitiative" covers "initiative". "federal popular initiative" covers all of
    "initiative" but only half of "federal_vote"."""
    parts = [p for p in option.lower().split("_") if p]
    words = set(value.split())
    # Short codes must be whole words: "rm" is inside "german" but means Romansh.
    hits = sum((p in value) if len(p) >= 4 else (p in words) for p in parts)
    return hits / len(parts) if parts else 0.0


def _ranked_choices(value: str, options: list[str]) -> list[tuple[str, float]]:
    import difflib

    def similarity(option: str) -> float:
        return difflib.SequenceMatcher(None, value, option.replace("_", " ").lower()).ratio()

    scored = [(o, _coverage(value, o), similarity(o)) for o in options]
    scored.sort(key=lambda s: (s[1], s[2]), reverse=True)  # similarity only breaks ties
    return [(o, cov if cov else (sim if sim >= 0.6 else 0.0)) for o, cov, sim in scored]


def repair_choice(
    arguments: dict[str, Any], message: str, required: set[str] | frozenset[str] = frozenset()
) -> tuple[dict[str, Any], str] | None:
    """Map the model's value to the closest allowed option, or drop it if it is optional
    (the model sent language "en" to a tool that only knows it, de, fr, rm)."""
    found = _invalid_choice(message)
    if not found:
        return None
    key, options = found
    value = normalize(str(arguments.get(key, "")))
    if not value or not options:
        return None
    best, score = _ranked_choices(value, options)[0]
    if not score:
        if key in required:
            return None
        rest = {k: v for k, v in arguments.items() if k != key}
        return rest, f"Dropped {key} {arguments.get(key)!r} (allowed: {', '.join(options)})"
    return {**arguments, key: best}, f"Corrected {key} {arguments.get(key)!r} to {best!r}"


def other_named_choices(arguments: dict[str, Any], message: str) -> list[dict[str, Any]]:
    """Arguments for every other allowed option the model's value names in full.

    "fakultatives Referendum und Volksinitiative" names two topics; Apertus got one lookup and
    invented the other figure. Each extra option becomes its own lookup.
    """
    found = _invalid_choice(message)
    if not found:
        return []
    key, options = found
    value = normalize(str(arguments.get(key, "")))
    ranked = [o for o, score in _ranked_choices(value, options) if score >= 1.0]
    return [{**arguments, key: o} for o in ranked[1:]]


def reroute_invented_coordinates(
    tool_name: str, arguments: dict[str, Any], question: str
) -> tuple[str, dict[str, Any], str] | None:
    """Users do not type coordinates. If the model made them up for a geo tool, the question
    is not really about a point on the map: send it to the full-text search instead."""
    if tool_name != "swiss_geo_context":
        return None
    coords = [arguments.get("latitude"), arguments.get("longitude")]
    if all(c is not None and str(c) in question for c in coords):
        return None
    return "search_knowledge", {"query": question}, "Rerouted: the question gives no coordinates"


def drop_named_optional(
    arguments: dict[str, Any], message: str, required: set[str] | frozenset[str] = frozenset()
) -> tuple[dict[str, Any], str] | None:
    """The tool complained about an optional argument by name ("... vat_rate standard or
    reduced"): drop it and let the tool use its default."""
    named = [k for k in arguments if k not in required and re.search(rf"\b{re.escape(k)}\b", message)]
    if not named:
        return None
    rest = {k: v for k, v in arguments.items() if k not in named}
    return rest, "Dropped " + ", ".join(f"{k} {arguments[k]!r}" for k in named) + " after a tool error"


def summarize_premiums(text: str) -> str:
    """One plain line on top of the premium JSON, so a small model quotes the right figure."""
    try:
        data = json.loads(text)
        rows = {p["accident_cover"]: p for p in data["premiums"]}
        without, with_ = rows["excluded"], rows["included"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return text
    line = (
        f"SUMMARY: {data.get('municipality')}, {data.get('age_group')}, franchise {data.get('franchise_chf')} CHF, "
        f"{data.get('year')}. Cheapest monthly premium WITHOUT accident cover: {without['minimum_premium']} CHF "
        f"({without['insurer']}, {without['model']}). Cheapest WITH accident cover: {with_['minimum_premium']} CHF "
        f"({with_['insurer']}, {with_['model']})."
    )
    return line + "\n" + text


# Text searches: their passages must share a topic word with the question (search_is_relevant).
SEARCH_TOOLS = {"search_knowledge", "swiss_housing_info"}
# Lookups that locate a place but answer nothing else on their own.
HELPER_TOOLS = {"swiss_geo_search", "swiss_geo_context"}
# Words that say nothing about the topic. Place names are removed separately.
STOPWORDS = set(
    """
    what when where which how much many does have with from that this your about there their would could
    should into than then them they were will just also only some more most other such very
    quel quelle quels quelles quand comment combien pour avec dans sont nous vous leur mais plus tres
    est-ce faut dois peux suis cette celui entre
    wann wie viel viele welche welcher welches warum wird werden sind habe haben kann muss soll ich mich
    mein meine eine einen einem einer oder aber auch noch nicht dann diese dieser dort hier fuer ueber
    quale quali quanto quanta quando come dove sono della delle degli dello nella nelle questo questa
    posso devo anche molto perche
    switzerland swiss schweiz schweizer suisse svizzera svizzero canton kanton cantone commune gemeinde comune
    """.split()
)


def topic_words(text: str, mcp_dir: Path) -> set[str]:
    """Content words of a question or query, without stopwords and place names."""
    words = set(normalize(text).split())
    return {
        w
        for w in words
        if len(w) >= 4 and w not in STOPWORDS and not w.isdigit()
        and canton_code(w) is None and lookup_bfs(w, mcp_dir) is None
    }


def search_is_relevant(question: str, arguments: dict[str, Any], result_text: str, mcp_dir: Path) -> bool:
    """False when no topic word of the question (or of the model's query) appears in the passages.

    Keyword search matched Bern's moving page for a question about Bern's waste collection on
    the place name alone, and both models then invented a schedule.
    """
    query = arguments.get("query") or arguments.get("question") or ""
    words = topic_words(question, mcp_dir) | topic_words(str(query), mcp_dir)
    if not words:
        return True  # nothing to check against
    passages = normalize(result_text)
    # Prefixes catch inflections: entsorge / Entsorgung, Karton / Kartons.
    return any(w[:6] in passages for w in words)


def places_in(text: str, mcp_dir: Path) -> list[tuple[str, int | None, str | None]]:
    """Places named in a text: (name as written, BFS code or None, canton code or None).

    Checks single words and two-word names ("La Chaux-de-Fonds" is handled by the normalizer).
    """
    words = [w for w in re.split(r"[^\w'-]+", text) if w]
    found: list[tuple[str, int | None, str | None]] = []
    seen: set[str] = set()
    for size in (2, 1):
        for i in range(len(words) - size + 1):
            name = " ".join(words[i : i + size])
            key = normalize(name)
            # Place names are written capitalised; "alle" in "alle elezioni" is not Alle (JU).
            if not name[:1].isupper():
                continue
            if len(key) < 4 or key in STOPWORDS or key in seen or any(key in s for s in seen):
                continue
            bfs = lookup_bfs(name, mcp_dir)
            canton = canton_code(name)
            if bfs or canton:
                found.append((name, bfs[0] if bfs else None, canton))
                seen.add(key)
    return found


def uncovered_places(question: str, lookups: list[tuple[str, dict[str, Any], str]], mcp_dir: Path) -> list[str]:
    """Places in the question that no lookup so far was about.

    Apertus said "done" after looking up Lausanne for "compare Lausanne and Geneva". Naming the
    missing place in the next-step prompt is cheap and does not depend on the model noticing.
    """
    covered_text = normalize(" ".join(json.dumps(args, ensure_ascii=False) + " " + text[:400] for _, args, text in lookups))
    codes = {str(v) for _, args, _ in lookups for v in args.values()}
    missing = []
    for name, bfs, canton in places_in(question, mcp_dir):
        key = normalize(name)
        alias = EXONYMS.get(key)
        if key in covered_text or (alias and alias in covered_text):
            continue
        if (bfs is not None and str(bfs) in codes) or (canton is not None and canton in codes):
            continue
        missing.append(name)
    return missing


# Tools whose main argument is a place, and how that argument is named (as the router sees it).
PLACE_ARGUMENTS = {
    PREMIUM_TOOL: ("municipality", "municipality"),
    "swiss_school_holidays": ("canton", "canton"),
    "get_driving_licence_exchange_info": ("canton", "canton"),
    "bfs_population": ("canton", "canton"),
}


def expand_for_place(tool: str, routed_arguments: dict[str, Any], place: str) -> tuple[str, dict[str, Any]] | None:
    """The same lookup for another place named in the question ("compare Lausanne and Geneva").

    Apertus noticed the missing place but left out which tool to call, so this does not rely on it.
    """
    if tool not in PLACE_ARGUMENTS:
        return None
    key, kind = PLACE_ARGUMENTS[tool]
    value: str | None = place
    if kind == "canton":
        value = canton_code(place)
        if value is None:
            return None
    arguments = {k: v for k, v in routed_arguments.items() if k not in {key, "municipality_code"}}
    arguments[key] = value
    return tool, arguments


# Words that join or compare parts of a question, in the four languages.
MULTI_PART_WORDS = set(
    """
    and or both compare comparison versus vs also plus
    et ou comparer compare comparaison ainsi aussi
    und oder sowie vergleich vergleiche vergleichen beide auch
    e ed o oppure confronta confrontare confronto entrambi anche
    """.split()
)


def may_have_several_parts(question: str, mcp_dir: Path) -> bool:
    """Cheap gate before asking the model for another lookup.

    The "anything missing?" call cost Apertus about 5 s per question, so single-part questions
    (most of them) skip it. Several places named, or a joining word, opens the gate.
    """
    words = set(normalize(question).split())
    return bool(words & MULTI_PART_WORDS) or len(places_in(question, mcp_dir)) > 1


def gap_is_covered(missing: str, lookups: list[tuple[str, dict[str, Any], str]], mcp_dir: Path) -> bool:
    """True when the results so far already contain most topic words of a gap the model named.

    Apertus called the Geneva premium "missing" after it had been looked up; chasing it added an
    irrelevant lookup and a note telling the answer that Geneva could not be checked.
    """
    words = topic_words(missing, mcp_dir)
    if not words:
        return False  # only places: uncovered_places decides
    covered = normalize(" ".join(json.dumps(args, ensure_ascii=False) + " " + text for _, args, text in lookups))
    found = sum(w[:6] in covered for w in words)
    return found / len(words) >= 0.6


# Word stems (accents removed) that name the topic of a place-based tool, in four languages.
PLACE_TOOL_STEMS = {
    "swiss_school_holidays": ("holiday", "vacanc", "ferien", "vacanz", "schulfer"),
    PREMIUM_TOOL: ("premium", "prime", "pramie", "praemie", "krankenkass", "assurance maladie", "cassa malati", "premio"),
    "get_driving_licence_exchange_info": ("licence", "license", "permis de conduire", "fuhrerausweis", "fuhrerschein", "patente"),
    "bfs_population": ("population", "bevolkerung", "einwohner", "popolazione", "abitanti", "habitants"),
}


def place_tool_for(question: str, used: set[str]) -> str | None:
    """A place-based tool whose topic the question names and that was not used yet.

    For "the rent rate and the autumn holidays in Ticino", Apertus said only "Ticino" was
    missing; the re-route then chose housing info and the answer invented holiday dates.
    """
    text = normalize(question)
    for tool, stems in PLACE_TOOL_STEMS.items():
        if tool not in used and any(stem in text for stem in stems):
            return tool
    return None
