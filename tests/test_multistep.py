"""Dynamic number of lookups per question, decided by the model, bounded by the orchestrator."""

import json

from gspaenli.orchestrator import Orchestrator
from gspaenli.router import parse_next_step
from tests.conftest import FakeLLM, FakeMCP

ALLOWED = {"swiss_health_insurance_premiums", "swiss_school_holidays", "search_knowledge"}


def _route(tool, arguments, lang="en"):
    return json.dumps({"tool": tool, "arguments": arguments, "lang": lang})


def _more(tool, arguments):
    return json.dumps({"done": False, "tool": tool, "arguments": arguments})


def test_parse_next_step():
    assert parse_next_step('{"done": true}', ALLOWED) is None
    assert parse_next_step(_more("swiss_school_holidays", {"canton": "GE"}), ALLOWED) == (
        "swiss_school_holidays",
        {"canton": "GE"},
    )
    # Anything unclear means stop, so a confused model cannot loop.
    for text in ["garbage", '{"done": false}', _more("crawl_federal_sources", {}), '{"tool": "search_knowledge"}']:
        assert parse_next_step(text, ALLOWED) is None


async def test_one_lookup_when_the_model_has_enough(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM([_route("swiss_health_insurance_premiums", {"municipality": "Lausanne"}), "Answer"])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("Premium in Lausanne?", "newcomer")
    assert len(mcp.calls) == 1 and len(llm.next_calls) == 0  # single-part: no extra model call
    assert turn.trace.stop_reason == "single-part question"


async def test_multi_part_question_asks_and_stops_when_done(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM([_route("swiss_health_insurance_premiums", {"municipality": "Lausanne"}), "Answer"])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Premium in Lausanne and the franchise options?", "newcomer"
    )
    assert len(mcp.calls) == 1 and len(llm.next_calls) == 1
    assert turn.trace.stop_reason == "the model had enough"


async def test_two_lookups_for_a_compound_question(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM(
        [_route("swiss_health_insurance_premiums", {"municipality": "Lausanne"}), "Both answers"],
        next_steps=[_more("swiss_health_insurance_premiums", {"municipality": "Genf"})],
    )
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Compare the minimum premium in Lausanne and Geneva for an adult", "newcomer"
    )
    assert [c[1]["municipality_code"] for c in mcp.calls] == [5586, 6621]
    t = turn.trace
    assert [s["tool"] for s in t.steps] == ["swiss_health_insurance_premiums"] * 2
    assert t.tool == "swiss_health_insurance_premiums" and t.arguments["municipality_code"] == 5586
    answer_prompt = llm.calls[1]["messages"][-1]["content"]
    assert "TOOL RESULT 1 of 2" in answer_prompt and "TOOL RESULT 2 of 2" in answer_prompt
    assert "next_step" in t.latency_s


async def test_lookups_are_capped(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM(
        [_route("search_knowledge", {"query": "a"}), "Answer"],
        next_steps=[_more("search_knowledge", {"query": q}) for q in ["b", "c", "d", "e"]],
    )
    turn = await Orchestrator(llm, mcp, personas, places_db, max_lookups=3).answer("x and y?", "fedlex_clerk")
    assert len(mcp.calls) == 3
    assert turn.trace.stop_reason == "reached the limit of 3 lookups"
    assert len(llm.next_calls) == 2  # no "next?" question after the last allowed lookup


async def test_repeated_lookup_stops_the_loop(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM(
        [_route("search_knowledge", {"query": "a"}), "Answer"],
        next_steps=[_more("search_knowledge", {"query": "a"})],
    )
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("x and y?", "fedlex_clerk")
    assert len(mcp.calls) == 1
    assert "already made" in turn.trace.stop_reason


async def test_one_failed_lookup_of_two_still_answers_with_the_good_one(personas, places_db):
    mcp = FakeMCP({"swiss_school_holidays": RuntimeError("timeout")})
    llm = FakeLLM(
        [_route("swiss_health_insurance_premiums", {"municipality": "Lausanne"}), "Partial answer"],
        next_steps=[_more("swiss_school_holidays", {"canton": "VD"})],
    )
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Premium in Lausanne and autumn holidays in Vaud?", "newcomer"
    )
    t = turn.trace
    assert t.tool_error is None  # one lookup worked
    assert t.steps[1]["error"] and "timeout" in t.steps[1]["error"]
    answer_prompt = llm.calls[1]["messages"][-1]["content"]
    assert "found nothing" not in llm.calls[1]["messages"][0]["content"]  # normal answer prompt
    assert "swiss_school_holidays" not in answer_prompt  # failed lookup not shown as data


async def test_all_lookups_failed_uses_the_no_data_prompt(personas, places_db):
    mcp = FakeMCP({"search_knowledge": RuntimeError("down")})
    llm = FakeLLM(
        [_route("search_knowledge", {"query": "a"}), "Not found"],
        next_steps=[_more("search_knowledge", {"query": "b"})],
    )
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("x and y?", "fedlex_clerk")
    assert len(turn.trace.steps) == 2 and turn.trace.tool_error
    assert "found nothing" in llm.calls[1]["messages"][0]["content"]


async def test_greeting_makes_no_lookup_and_no_next_step_call(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM([_route("none", {}), "Hi!"])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("Hello!", "buenzli")
    assert mcp.calls == [] and llm.next_calls == [] and turn.trace.steps == []


async def test_next_step_prompt_names_places_not_yet_looked_up(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM([_route("swiss_health_insurance_premiums", {"municipality": "Lausanne"}), "Answer"])
    await Orchestrator(llm, mcp, personas, places_db).answer(
        "Compare the minimum premium in Lausanne and Geneva for an adult", "newcomer"
    )
    prompt = llm.next_calls[0]["messages"][-1]["content"]
    assert "Places in the question with no lookup yet: Geneva" in prompt
    assert "missing" in llm.next_calls[0]["json_schema"]["required"]


def test_uncovered_places(places_db):
    from gspaenli.resolver import uncovered_places

    q = "Compare the minimum premium in Lausanne and Geneva"
    done = [("swiss_health_insurance_premiums", {"municipality_code": 5586}, '{"municipality": "Lausanne"}')]
    assert uncovered_places(q, done, places_db) == ["Geneva"]
    done.append(("swiss_health_insurance_premiums", {"municipality_code": 6621}, '{"municipality": "Genève"}'))
    assert uncovered_places(q, done, places_db) == []


async def test_missing_place_is_added_when_the_model_names_no_tool(personas, places_db):
    # Apertus replied {"missing": "Geneva", "done": false} without a tool.
    mcp = FakeMCP()
    llm = FakeLLM(
        [_route("swiss_health_insurance_premiums", {"municipality": "Lausanne"}), "Both"],
        next_steps=['{"missing": "Geneva", "done": false}', '{"missing": "nothing", "done": true}'],
    )
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Compare the minimum premium in Lausanne and Geneva for an adult", "newcomer"
    )
    assert [c[1]["municipality_code"] for c in mcp.calls] == [5586, 6621]
    assert any("Added a lookup for Geneva" in n for n in turn.trace.notes)
    assert not any("No lookup was made" in n for n in turn.trace.notes)


async def test_place_left_uncovered_is_flagged_to_the_answer(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM([_route("swiss_health_insurance_premiums", {"municipality": "Lausanne"}), "Answer"])
    turn = await Orchestrator(llm, mcp, personas, places_db, max_lookups=1).answer(
        "Compare the minimum premium in Lausanne and Geneva for an adult", "newcomer"
    )
    notes = " ".join(turn.trace.notes)
    assert "No lookup was made for" in notes and "Geneva" in notes
    assert "No lookup was made for" in llm.calls[1]["messages"][-1]["content"]


def test_expand_for_place():
    from gspaenli.resolver import expand_for_place

    assert expand_for_place("swiss_health_insurance_premiums", {"municipality": "Lausanne", "age": 40}, "Geneva") == (
        "swiss_health_insurance_premiums",
        {"age": 40, "municipality": "Geneva"},
    )
    assert expand_for_place("swiss_school_holidays", {"canton": "VD"}, "Genève") == ("swiss_school_holidays", {"canton": "GE"})
    assert expand_for_place("swiss_school_holidays", {"canton": "VD"}, "Atlantis") is None
    assert expand_for_place("search_knowledge", {"query": "x"}, "Geneva") is None


async def test_value_naming_two_topics_gets_two_lookups(personas, places_db):
    # Apertus sent topic "fakultatives Referendum und Volksinitiative", got one lookup and invented
    # the initiative figure.
    from tests.test_guards import INVALID_TOPIC, SequenceMCP

    answered = {"status": "answered", "source_url": "https://www.bk.admin.ch/x"}
    mcp = SequenceMCP({"swiss_federal_political_rights": [INVALID_TOPIC, answered, answered]})
    topic = "fakultatives Referendum und Volksinitiative"
    llm = FakeLLM([_route("swiss_federal_political_rights", {"topic": topic}, "de"), "Beide"])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Wie viele Unterschriften braucht ein fakultatives Referendum und wie viele eine Volksinitiative?",
        "fedlex_clerk",
    )
    assert sorted(c[1]["topic"] for c in mcp.calls[1:]) == ["initiative", "referendum"]
    assert len(turn.trace.steps) == 2
    assert any("Also looked up" in n for n in turn.trace.notes)


async def test_lookup_resolving_to_one_already_made_is_skipped(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM(
        [_route("swiss_health_insurance_premiums", {"municipality": "Lausanne"}), "Answer"],
        next_steps=[_more("swiss_health_insurance_premiums", {"municipality": "Losanna"})],
    )
    turn = await Orchestrator(llm, mcp, personas, places_db).answer("Premium in Lausanne and more?", "newcomer")
    assert len(mcp.calls) == 1  # Losanna resolves to the same municipality: no second MCP call
    assert "already made" in turn.trace.stop_reason


async def test_still_missing_topic_is_routed(personas, places_db):
    # A gap with no place: the router runs on the missing part alone, with unused tools only.
    mcp = FakeMCP()
    mcp.tools["swiss_reference_interest_rate"] = mcp.tools["search_knowledge"].__class__(
        "swiss_reference_interest_rate", "", {"type": "object", "properties": {}}
    )
    mcp.tools["swiss_import_parcel_vat"] = mcp.tools["search_knowledge"].__class__(
        "swiss_import_parcel_vat", "", {"type": "object", "properties": {"goods_value_chf": {"type": "number"}}}
    )
    llm = FakeLLM(
        [
            _route("swiss_reference_interest_rate", {}, "it"),
            _route("swiss_import_parcel_vat", {"goods_value_chf": 200}, "it"),
            "Risposta",
        ],
        next_steps=['{"missing": "IVA su un pacco da 200 franchi", "done": false}'],
    )
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Qual è il tasso di riferimento e quanta IVA pago su un pacco da 200 franchi?", "fedlex_clerk"
    )
    assert [c[0] for c in mcp.calls] == ["swiss_reference_interest_rate", "swiss_import_parcel_vat"]
    assert any("still missing" in n for n in turn.trace.notes)
    route_again = llm.calls[1]["messages"][-1]["content"]
    assert "IVA su un pacco da 200 franchi" in route_again and "tasso" not in route_again  # focused sub-question
    assert "swiss_reference_interest_rate" not in llm.calls[1]["messages"][0]["content"]  # used tool not offered


def test_parse_missing():
    from gspaenli.router import parse_missing

    assert parse_missing('{"missing": "Geneva", "done": false}') == "Geneva"
    assert parse_missing('{"missing": "nothing", "done": false}') is None
    assert parse_missing('{"missing": "Geneva", "done": true}') is None
    assert parse_missing("garbage") is None



def test_multi_part_gate(places_db):
    from gspaenli.resolver import may_have_several_parts

    assert not may_have_several_parts("What is the minimum premium in Lausanne for an adult?", places_db)
    assert not may_have_several_parts("Quand sont les vacances d'automne dans le canton de Vaud ?", places_db)
    assert may_have_several_parts("Compare the premium in Lausanne and Geneva", places_db)
    assert may_have_several_parts("Prime à Genève, et vacances d'automne ?", places_db)
    assert may_have_several_parts("Referendum und Volksinitiative?", places_db)
    assert may_have_several_parts("Tasso di riferimento e vacanze in Ticino?", places_db)


async def test_still_missing_reroute_offers_only_unused_tools(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM(
        [
            _route("swiss_health_insurance_premiums", {"municipality": "Genf"}, "fr"),
            _route("swiss_school_holidays", {"canton": "GE"}, "fr"),
            "Réponse",
        ],
        next_steps=['{"missing": "vacances d\'automne à Genève", "done": false}'],
    )
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Prime minimale à Genève pour un adulte, et quand sont les vacances d'automne à Genève ?", "newcomer"
    )
    reroute_system = llm.calls[1]["messages"][0]["content"]
    assert "swiss_health_insurance_premiums" not in reroute_system  # already used
    assert [c[0] for c in mcp.calls] == ["swiss_health_insurance_premiums", "swiss_school_holidays"]
    assert not any(n.startswith("Not looked up") for n in turn.trace.notes)


async def test_gap_never_looked_up_is_flagged_to_the_answer(personas, places_db):
    mcp = FakeMCP()
    llm = FakeLLM(
        [
            _route("swiss_health_insurance_premiums", {"municipality": "Genf"}, "fr"),
            _route("none", {}, "fr"),  # no unused tool fits the gap
            "Réponse",
        ],
        next_steps=['{"missing": "vacances d\'automne à Genève", "done": false}'],
    )
    mcp.tools = {k: v for k, v in mcp.tools.items() if k in {"swiss_health_insurance_premiums", "search_knowledge"}}
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Prime minimale à Genève pour un adulte, et quand sont les vacances d'automne à Genève ?", "newcomer"
    )
    assert any("Not looked up: vacances d'automne à Genève" in n for n in turn.trace.notes)
    assert "Not looked up" in llm.calls[-1]["messages"][-1]["content"]


async def test_location_lookup_alone_is_no_data_for_a_topic_question(personas, places_db):
    mcp = FakeMCP({"swiss_geo_search": {"status": "answered", "locations": [{"name": "Bern"}]}})
    llm = FakeLLM([_route("swiss_geo_search", {"query": "Bern"}, "de"), "Nicht gefunden"])
    turn = await Orchestrator(llm, mcp, personas, places_db).answer(
        "Wann ist in Bern die Kehrichtabfuhr und wie entsorge ich Karton?", "buenzli"
    )
    assert "found nothing" in llm.calls[-1]["messages"][0]["content"]


def test_short_codes_must_match_whole_words():
    from gspaenli.resolver import repair_choice

    msg = (
        "1 validation error for call[x]\nlanguage\n"
        "  Input should be 'it', 'de', 'fr' or 'rm' [type=literal_error, input_value='German']"
    )
    args, note = repair_choice({"language": "German"}, msg)
    assert args == {} and "Dropped language" in note  # not "rm", which is inside "German"



def test_only_capitalised_words_are_places(places_db):
    from gspaenli.resolver import places_in

    assert places_in("Compare Lausanne and Geneva", places_db) == [("Lausanne", 5586, None), ("Geneva", 6621, "GE")]
    assert places_in("compare lausanne and geneva", places_db) == []


def test_missing_that_restates_the_question_is_ignored():
    from gspaenli.orchestrator import _restates

    q = "Compare the minimum health insurance premium for an adult in Lausanne and in Geneva."
    assert _restates("Compare the minimum health insurance premium for an adult in Lausanne and in Geneva", q)
    assert not _restates("Geneva", q)
    assert not _restates("autumn holidays in Ticino", q)


def test_gap_already_covered_by_results(places_db):
    from gspaenli.resolver import gap_is_covered

    premium = '{"municipality": "Genève", "age_group": "adult_26_plus", "premiums": [{"minimum_premium": "562.0"}]}'
    done = [("swiss_health_insurance_premiums", {"municipality_code": 6621}, premium)]
    assert gap_is_covered("minimum health insurance premium for an adult in Geneva", done, places_db)
    assert not gap_is_covered("vacanze autunnali in Ticino", done, places_db)


async def test_uncovered_place_gets_the_place_tool_the_question_names(personas, places_db):
    # Apertus said only "Ticino" was missing after the rent rate lookup.
    mcp = FakeMCP()
    mcp.tools["swiss_reference_interest_rate"] = mcp.tools["search_knowledge"].__class__(
        "swiss_reference_interest_rate", "", {"type": "object", "properties": {}}
    )
    llm = FakeLLM(
        [_route("swiss_reference_interest_rate", {}, "it"), "Risposta"],
        next_steps=['{"missing": "Ticino", "done": false}'],
    )
    await Orchestrator(llm, mcp, personas, places_db).answer(
        "Qual è il tasso di riferimento per gli affitti e quando sono le vacanze autunnali in Ticino?", "buenzli"
    )
    assert mcp.calls[1] == ("swiss_school_holidays", {"canton": "TI"})


def test_place_tool_for():
    from gspaenli.resolver import place_tool_for

    assert place_tool_for("tasso di riferimento e vacanze autunnali in Ticino", set()) == "swiss_school_holidays"
    assert place_tool_for("Prämie in Genf und Ferien", {"swiss_school_holidays"}) == "swiss_health_insurance_premiums"
    assert place_tool_for("reference interest rate", set()) is None
