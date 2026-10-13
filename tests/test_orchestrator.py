import json

from gspaenli.orchestrator import Orchestrator
from tests.conftest import FakeLLM, FakeMCP

PREMIUM_RESULT = {
    "status": "answered",
    "municipality": "Lausanne",
    "premiums": [{"minimum_premium": "531.7", "source_url": "https://kvgmapper26.github.io/maps/a.html"}],
    "official_reference": "https://www.priminfo.admin.ch/de/praemien",
}


def _route(tool, arguments, lang):
    return json.dumps({"tool": tool, "arguments": arguments, "lang": lang})


async def test_full_turn_with_premium_tool(personas, places_db):
    mcp = FakeMCP({"swiss_health_insurance_premiums": PREMIUM_RESULT})
    llm = FakeLLM([_route("swiss_health_insurance_premiums", {"municipality": "Lausanne"}, "fr"), "Réponse"])
    orch = Orchestrator(llm, mcp, personas, places_db)

    turn = await orch.answer("Quelle est la prime minimale à Lausanne pour un adulte ?", "newcomer")

    assert turn.answer == "Réponse"
    assert mcp.calls == [("swiss_health_insurance_premiums", {"age": 30, "municipality_code": 5586, "franchise": 300})]
    t = turn.trace
    assert t.tool == "swiss_health_insurance_premiums" and t.lang == "fr"
    assert t.sources == ["https://kvgmapper26.github.io/maps/a.html", "https://www.priminfo.admin.ch/de/praemien"]
    assert t.model_id == "fake/model"
    assert set(t.latency_s) == {"route", "tool", "answer", "total"}  # single-part: no next-step call
    assert len(t.steps) == 1 and t.stop_reason == "single-part question"
    answer_prompt = llm.calls[1]["messages"][-1]["content"]
    assert "531.7" in answer_prompt and "Answer in French" in answer_prompt
    assert llm.calls[1]["json_schema"] is None


async def test_no_tool_turn(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM([_route("none", {}, "it"), "Ciao!"])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("Ciao, come stai?", "buenzli")
    assert turn.answer == "Ciao!" and mcp.calls == [] and turn.trace.tool is None
    assert "Answer in Italian" in llm.calls[1]["messages"][-1]["content"]


async def test_tool_exception_becomes_missing_data(personas, places_db):
    mcp = FakeMCP({"search_knowledge": RuntimeError("pipe closed")})
    llm = FakeLLM([_route("search_knowledge", {"query": "x"}, "en"), "Sorry, no data."])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("x?", "fedlex_clerk")
    assert "pipe closed" in turn.trace.tool_error
    assert "found nothing" in llm.calls[1]["messages"][0]["content"]


async def test_unresolvable_place_skips_tool_call(personas, places_db):
    mcp = FakeMCP({"swiss_geo_search": {"locations": []}})
    llm = FakeLLM([_route("swiss_health_insurance_premiums", {"municipality": "Atlantis"}, "en"), "Unknown place."])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("Premium in Atlantis?", "newcomer")
    assert [c[0] for c in mcp.calls] == ["swiss_geo_search"]
    assert "Atlantis" in turn.trace.tool_error


async def test_router_failure_falls_back_to_search_knowledge(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM(["?", "??", "Answer"])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("Abfall in Bern?", "buenzli")
    assert turn.trace.fallback and mcp.calls == [("search_knowledge", {"query": "Abfall in Bern?"})]
