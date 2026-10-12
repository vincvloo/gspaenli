import json

import pytest

from gspaenli.resolver import (
    PREMIUM_TOOL,
    canton_code,
    coerce_arguments,
    lookup_bfs,
    normalize,
    resolve,
    router_tools,
)
from tests.conftest import FakeMCP


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Lausanne", 5586), ("LAUSANNE", 5586), ("Losanna", 5586), ("Genève", 6621), ("Genf", 6621),
     ("Geneva", 6621), ("St. Gallen", 3203), ("Sankt Gallen", 3203)],
)
def test_lookup_bfs_handles_accents_case_and_exonyms(places_db, name, expected):
    assert lookup_bfs(name, places_db)[0] == expected


def test_lookup_bfs_unknown_and_missing_db(places_db, tmp_path):
    assert lookup_bfs("Atlantis", places_db) is None
    assert lookup_bfs("Lausanne", tmp_path / "nowhere") is None


def test_normalize():
    assert normalize("La Chaux-de-Fonds") == "la chaux de fonds"
    assert normalize("Biel/Bienne") == "biel bienne"


@pytest.mark.parametrize(
    ("value", "code"), [("VD", "VD"), ("ch-zh", "ZH"), ("Vaud", "VD"), ("Waadt", "VD"), ("Ginevra", "GE"), ("Tessin", "TI"), ("Atlantis", None)]
)
def test_canton_code(value, code):
    assert canton_code(value) == code


async def test_premium_with_defaults_states_assumptions(places_db):
    mcp = FakeMCP()
    res = await resolve(PREMIUM_TOOL, {"municipality": "Lausanne"}, mcp.tools, mcp, places_db)
    assert res.error is None
    assert res.arguments == {"age": 30, "municipality_code": 5586, "franchise": 300}
    assert any("aged 30" in n for n in res.notes)
    assert any("300 CHF franchise" in n for n in res.notes)
    assert mcp.calls == []  # local lookup, no extra MCP calls


async def test_premium_keeps_user_values_and_fixes_bad_franchise(places_db):
    mcp = FakeMCP()
    res = await resolve(PREMIUM_TOOL, {"municipality": "Genf", "age": "22", "franchise": 2500}, mcp.tools, mcp, places_db)
    assert res.arguments == {"age": 22, "municipality_code": 6621, "franchise": 2500}
    res = await resolve(PREMIUM_TOOL, {"municipality": "Genf", "age": 40, "franchise": 750}, mcp.tools, mcp, places_db)
    assert res.arguments["franchise"] == 300


async def test_premium_accepts_name_in_code_field(places_db):
    mcp = FakeMCP()
    res = await resolve(PREMIUM_TOOL, {"municipality_code": "Lausanne"}, mcp.tools, mcp, places_db)
    assert res.arguments["municipality_code"] == 5586


async def test_premium_falls_back_to_geo_tools(places_db):
    mcp = FakeMCP(
        {
            "swiss_geo_search": {"locations": [{"latitude": 46.2, "longitude": 6.1}]},
            "swiss_geo_context": {"municipality": {"name": "Carouge (GE)", "bfs_number": 6608}},
        }
    )
    res = await resolve(PREMIUM_TOOL, {"municipality": "Carouge"}, mcp.tools, mcp, places_db)
    assert res.arguments["municipality_code"] == 6608
    assert [c[0] for c in mcp.calls] == ["swiss_geo_search", "swiss_geo_context"]
    assert len(res.extra_calls) == 2


async def test_premium_unknown_place_is_an_error(places_db):
    mcp = FakeMCP({"swiss_geo_search": {"locations": []}})
    res = await resolve(PREMIUM_TOOL, {"municipality": "Atlantis"}, mcp.tools, mcp, places_db)
    assert res.error and "Atlantis" in res.error


async def test_canton_names_become_codes_for_other_tools(places_db):
    mcp = FakeMCP()
    res = await resolve(
        "swiss_school_holidays", {"canton": "Vaud", "school_year": "2026", "holiday_type": "automne"}, mcp.tools, mcp, places_db
    )
    assert res.arguments == {"canton": "VD", "school_year": 2026}


def test_coerce_drops_unknown_keys():
    tool = FakeMCP().tools["search_knowledge"]
    assert coerce_arguments({"query": "x", "limit": "3", "bogus": 1}, tool) == {"query": "x", "limit": 3}


def test_router_sees_municipality_name_not_code():
    shown = router_tools(FakeMCP().tools)[PREMIUM_TOOL]
    props = shown.input_schema["properties"]
    assert "municipality" in props and "municipality_code" not in props
    assert json.dumps(shown.input_schema)
