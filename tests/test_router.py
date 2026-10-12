import pytest

from gspaenli.router import RouteError, parse_route, route, route_schema
from tests.conftest import FakeLLM, FakeMCP

ALLOWED = {"search_knowledge", "swiss_health_insurance_premiums"}


def test_parses_plain_json():
    r = parse_route('{"tool": "search_knowledge", "arguments": {"query": "x"}, "lang": "fr"}', ALLOWED, "q")
    assert (r.tool, r.arguments, r.lang) == ("search_knowledge", {"query": "x"}, "fr")


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"tool": "search_knowledge", "arguments": {}, "lang": "de"}\n```',
        'Sure! Here is the JSON: {"tool": "search_knowledge", "arguments": {}, "lang": "de"} Hope it helps.',
        '{"tool": "search_knowledge", "arguments": {}, "lang": "de"}\n{"second": 1}',
    ],
)
def test_tolerates_fences_and_prose(text):
    assert parse_route(text, ALLOWED, "q").tool == "search_knowledge"


def test_none_means_no_tool():
    r = parse_route('{"tool": "none", "arguments": {}, "lang": "en"}', ALLOWED, "hi")
    assert r.tool is None


def test_missing_or_bad_lang_is_guessed_from_question():
    r = parse_route('{"tool": "none", "arguments": {}}', ALLOWED, "Quelle est la prime pour une famille ?")
    assert r.lang == "fr"
    r = parse_route('{"tool": "none", "arguments": {}, "lang": "es"}', ALLOWED, "Wie viel kostet das für mich?")
    assert r.lang == "de"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no json here", "no JSON"),
        ('{"tool": "crawl_federal_sources", "arguments": {}}', "unknown tool"),
        ('{"tool": "search_knowledge", "arguments": "x"}', "arguments"),
        ('{"tool": 3}', "string"),
        ('{"tool": "search_knowledge", "arguments": {', "invalid JSON"),
        ("[1, 2]", "no JSON"),
    ],
)
def test_rejects_invalid_routes(text, message):
    with pytest.raises(RouteError, match=message):
        parse_route(text, ALLOWED, "q")


def test_schema_enumerates_tools_and_none():
    schema = route_schema(["a", "b"])
    assert schema["properties"]["tool"]["enum"] == ["a", "b", "none"]
    assert schema["properties"]["lang"]["enum"] == ["en", "fr", "de", "it"]


async def test_route_succeeds_first_try(personas):
    llm = FakeLLM(['{"tool": "search_knowledge", "arguments": {"query": "x"}, "lang": "en"}'])
    r = await route(llm, "q", personas["newcomer"], FakeMCP().tools)
    assert r.tool == "search_knowledge" and r.attempts == 1 and not r.fallback
    assert llm.calls[0]["json_schema"] is not None


async def test_route_retries_once_with_error_feedback(personas):
    llm = FakeLLM(["nope", '{"tool": "swiss_school_holidays", "arguments": {"canton": "VD"}, "lang": "fr"}'])
    r = await route(llm, "Vacances ?", personas["buenzli"], FakeMCP().tools)
    assert r.tool == "swiss_school_holidays" and r.attempts == 2
    retry_messages = llm.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "nope"}
    assert "not valid" in retry_messages[-1]["content"]


async def test_route_falls_back_to_search_knowledge_after_two_failures(personas):
    llm = FakeLLM(["garbage", '{"tool": "made_up_tool", "arguments": {}}'])
    question = "Wie funktioniert die Abfallentsorgung in Bern?"
    r = await route(llm, question, personas["buenzli"], FakeMCP().tools)
    assert r.fallback and r.tool == "search_knowledge"
    assert r.arguments == {"query": question}
    assert r.lang == "de"
    assert len(llm.calls) == 2
