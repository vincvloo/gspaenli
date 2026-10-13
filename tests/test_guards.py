"""Guards added after the first eval run (see evals/README.md)."""

import json

from gspaenli.mcp_client import ToolInfo, ToolResult, extract_sources
from gspaenli.orchestrator import Orchestrator
from gspaenli.resolver import repair_choice, reroute_invented_coordinates, result_problem
from tests.conftest import FakeLLM, FakeMCP

INVALID_TOPIC = {
    "status": "invalid_input",
    "message": "topic must be one of: initiative, referendum, national_council, petition, federal_vote.",
}


def test_result_problem():
    assert "must be one of" in result_problem(json.dumps(INVALID_TOPIC))
    assert result_problem(json.dumps({"status": "answered", "x": 1})) is None
    assert result_problem(json.dumps({"no_status": True})) is None
    assert result_problem("School holidays for VD ...") is None


def test_repair_choice_maps_free_text_to_allowed_option():
    msg = INVALID_TOPIC["message"]
    assert repair_choice({"topic": "federal popular initiative"}, msg)[0] == {"topic": "initiative"}
    assert repair_choice({"topic": "National Council elections"}, msg)[0] == {"topic": "national_council"}
    assert repair_choice({"topic": "federal vote"}, msg)[0] == {"topic": "federal_vote"}
    assert repair_choice({"topic": "zzzz"}, msg, required={"topic"}) is None
    assert repair_choice({"topic": "x"}, "some other error") is None


PYDANTIC_LANGUAGE = (
    "1 validation error for call[swiss_reference_interest_rate]\nlanguage\n"
    "  Input should be 'it', 'de', 'fr' or 'rm' [type=literal_error, input_value='en', input_type=str]"
)
PYDANTIC_FACT = (
    "1 validation error for call[get_driving_licence_exchange_info]\nfact_type\n"
    "  Input should be 'all', 'fee', 'deadline' or 'requirement' [type=literal_error, input_value='fees']"
)


def test_repair_pydantic_literal_errors():
    args, note = repair_choice({"canton": "ZH", "fact_type": "fees"}, PYDANTIC_FACT, required={"canton"})
    assert args == {"canton": "ZH", "fact_type": "fee"} and "Corrected" in note
    # No sensible mapping for an optional argument: drop it and let the tool use its default.
    args, note = repair_choice({"language": "en"}, PYDANTIC_LANGUAGE)
    assert args == {} and "Dropped language" in note


async def test_invented_dates_are_dropped(places_db):
    from gspaenli.resolver import resolve

    mcp = FakeMCP()
    mcp.tools["swiss_reference_interest_rate"] = ToolInfo(
        "swiss_reference_interest_rate", "", {"type": "object", "properties": {"as_of": {"type": "string"}}}
    )
    res = await resolve(
        "swiss_reference_interest_rate", {"as_of": "2023-10-10"}, mcp.tools, mcp, places_db,
        "What is the current reference interest rate for rents?",
    )
    assert res.arguments == {} and "no date in the question" in res.notes[0]
    res = await resolve(
        "swiss_reference_interest_rate", {"as_of": "2025-06-30"}, mcp.tools, mcp, places_db,
        "What was the reference rate in June 2025?",
    )
    assert res.arguments == {"as_of": "2025-06-30"}


def test_reroute_invented_coordinates():
    q = "Wann ist in Bern die Kehrichtabfuhr?"
    tool, args, note = reroute_invented_coordinates("swiss_geo_context", {"latitude": 46.94, "longitude": 7.44}, q)
    assert tool == "search_knowledge" and args == {"query": q} and "coordinates" in note
    q2 = "Which municipality is at 46.94, 7.44?"
    assert reroute_invented_coordinates("swiss_geo_context", {"latitude": 46.94, "longitude": 7.44}, q2) is None
    assert reroute_invented_coordinates("search_knowledge", {"query": "x"}, q) is None


class SequenceMCP(FakeMCP):
    """Returns queued payloads per tool, in order."""

    def __init__(self, queue):
        super().__init__()
        self.tools["swiss_federal_political_rights"] = ToolInfo(
            "swiss_federal_political_rights", "", {"type": "object", "properties": {"topic": {"type": "string"}}}
        )
        self.queue = queue

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        text = json.dumps(self.queue[name].pop(0))
        return ToolResult(name=name, arguments=arguments, text=text, sources=extract_sources(text))


def _route(tool, arguments, lang="en"):
    return json.dumps({"tool": tool, "arguments": arguments, "lang": lang})


async def test_invalid_topic_is_repaired_and_retried(personas, places_db):
    answered = {"status": "answered", "rule": "100 000 signatures in 18 months", "source_url": "https://www.bk.admin.ch/x"}
    mcp = SequenceMCP({"swiss_federal_political_rights": [INVALID_TOPIC, answered]})
    llm = FakeLLM([_route("swiss_federal_political_rights", {"topic": "federal popular initiative"}), "100,000."])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "How many signatures does a federal popular initiative need?", "fedlex_clerk"
    )
    assert [c[1] for c in mcp.calls] == [{"topic": "federal popular initiative"}, {"topic": "initiative"}]
    assert turn.trace.arguments == {"topic": "initiative"} and turn.trace.tool_error is None
    assert any("Corrected topic" in n for n in turn.trace.notes)
    assert turn.trace.sources == ["https://www.bk.admin.ch/x"]


async def test_unrepairable_non_answer_is_flagged_as_missing_data(personas, places_db):
    mcp = SequenceMCP({"swiss_federal_political_rights": [{"status": "not_found", "message": "no rule found"}]})
    llm = FakeLLM([_route("swiss_federal_political_rights", {"topic": "initiative"}), "No data."])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("Initiative?", "fedlex_clerk")
    assert turn.trace.tool_error == "no rule found"
    system = llm.calls[1]["messages"][0]["content"]
    assert "found nothing" in system and "Not even ones you know" in system
    assert "no rule found" not in str(llm.calls[1]["messages"])  # no tool output to improvise from


async def test_invented_coordinates_go_to_full_text_search(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM([_route("swiss_geo_context", {"latitude": 46.9481, "longitude": 7.4478}, "de"), "Antwort"])
    q = "Wann ist in Bern die Kehrichtabfuhr und wie entsorge ich Karton?"
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(q, "buenzli")
    assert mcp.calls == [("search_knowledge", {"query": q})]
    assert turn.trace.tool == "search_knowledge"


async def test_prompt_label_is_removed_from_answers(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM([_route("search_knowledge", {"query": "x"}), "Quelle: swiss_import_parcel_vat (TOOL RESULT)."])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("x?", "fedlex_clerk")
    assert "TOOL RESULT" not in turn.answer


def test_null_strings_from_the_router_are_dropped():
    from gspaenli.resolver import coerce_arguments

    tool = ToolInfo("t", "", {"properties": {"canton": {"type": "string"}, "municipality": {"type": "string"}}})
    assert coerce_arguments({"canton": "VD", "municipality": "None"}, tool) == {"canton": "VD"}


def test_optional_argument_named_in_an_error_is_dropped():
    from gspaenli.resolver import drop_named_optional

    msg = "Use non-negative CHF amounts and vat_rate standard or reduced."
    args, note = drop_named_optional({"goods_value_chf": 200.0, "vat_rate": "8.1%"}, msg, {"goods_value_chf"})
    assert args == {"goods_value_chf": 200.0} and "vat_rate" in note
    assert drop_named_optional({"goods_value_chf": -1}, "Use non-negative CHF amounts", {"goods_value_chf"}) is None


def test_premium_summary_names_the_cheapest_figures():
    from gspaenli.resolver import summarize_premiums

    data = {
        "municipality": "Zürich", "age_group": "young_19_25", "franchise_chf": 2500, "year": 2026,
        "premiums": [
            {"accident_cover": "excluded", "minimum_premium": "225.9", "insurer": "Assura", "model": "Qualimed"},
            {"accident_cover": "included", "minimum_premium": "242.9", "insurer": "Assura", "model": "Qualimed"},
        ],
    }
    first_line = summarize_premiums(json.dumps(data)).splitlines()[0]
    assert "WITHOUT accident cover: 225.9 CHF" in first_line and "WITH accident cover: 242.9 CHF" in first_line
    assert summarize_premiums("not json") == "not json"


def test_router_sees_eu_countries_for_permits():
    from gspaenli.resolver import PERMIT_TOOL, router_tools

    tools = {PERMIT_TOOL: ToolInfo(PERMIT_TOOL, "Permit guidance.", {"properties": {"group": {"type": "string"}}})}
    assert "Germany" in router_tools(tools)[PERMIT_TOOL].description



async def test_irrelevant_search_results_become_no_data(personas, places_db):
    # Regression: keyword search matched only the place name and both models invented a schedule.
    passages = {"method": "keyword_fallback", "results": [
        {"url": "https://www.bern.ch/themen/zuzug-umzug-wegzug/", "passage": "Willkommen in der Stadt Bern. Umzug melden."}
    ]}
    mcp = FakeMCP({"search_knowledge": passages})
    llm = FakeLLM([_route("search_knowledge", {"query": "Kehrichtabfuhr Bern Karton"}, "de"), "Nicht gefunden."])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Wann ist in Bern die Kehrichtabfuhr und wie entsorge ich Karton?", "buenzli"
    )
    assert "matches the topic" in turn.trace.tool_error
    assert turn.trace.sources == []
    assert "found nothing" in llm.calls[1]["messages"][0]["content"]


async def test_relevant_search_results_are_kept(personas, places_db):
    passages = {"results": [{"url": "https://www.regiunebvm.ch/de/kehricht/", "passage": "Kehricht und Karton werden ..."}]}
    mcp = FakeMCP({"search_knowledge": passages})
    llm = FakeLLM([_route("search_knowledge", {"query": "Kehricht Karton Scuol"}, "de"), "Antwort"])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("Wann ist die Kehrichtabfuhr in Scuol?", "buenzli")
    assert turn.trace.tool_error is None and turn.trace.sources == ["https://www.regiunebvm.ch/de/kehricht/"]


async def test_no_tool_for_a_real_question_falls_back_to_search(personas, places_db):
    mcp = FakeMCP({"search_knowledge": {"results": [{"url": "https://x", "passage": "Umzug melden"}]}})
    llm = FakeLLM([_route("none", {}, "de"), "Nicht gefunden."])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Wann ist in Bern die Kehrichtabfuhr und wie entsorge ich Karton?", "buenzli"
    )
    assert mcp.calls[0][0] == "search_knowledge"
    assert turn.trace.tool_error  # nothing relevant, so the honest no-data answer


async def test_greetings_still_need_no_tool(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM([_route("none", {}, "it"), "Ciao!"])
    await Orchestrator(llm, mcp, personas, places_db).answer("Ciao, come stai?", "buenzli")
    assert mcp.calls == []
