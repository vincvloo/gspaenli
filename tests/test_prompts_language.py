import pytest

from gspaenli.prompts import (
    LANG_NAMES,
    REPLY_LANGUAGE_RULE,
    build_answer_messages,
    build_next_step_messages,
    build_route_messages,
    guess_language,
)
from tests.conftest import FakeMCP


def _system(messages):
    return messages[0]["content"]


@pytest.mark.parametrize("persona_id", ["newcomer", "fedlex_clerk", "buenzli"])
@pytest.mark.parametrize("lang", list(LANG_NAMES))
def test_reply_language_rule_in_every_prompt(personas, persona_id, lang):
    persona = personas[persona_id]
    route_msgs = build_route_messages("question", persona, FakeMCP().tools)
    assert REPLY_LANGUAGE_RULE in _system(route_msgs)

    one = [("search_knowledge", {}, "{}")]
    two = [("swiss_health_insurance_premiums", {}, "{}"), ("swiss_school_holidays", {}, "VD")]
    for lookups in ([], one, two):
        answer_msgs = build_answer_messages("question", persona, lang, lookups, [])
        assert REPLY_LANGUAGE_RULE in _system(answer_msgs)
        last = answer_msgs[-1]["content"]
        assert f"Answer in {LANG_NAMES[lang]}" in last
        assert last.rstrip().endswith(REPLY_LANGUAGE_RULE)  # the language instruction comes last
    next_msgs = build_next_step_messages("question", FakeMCP().tools, one)
    assert REPLY_LANGUAGE_RULE in _system(next_msgs)


def test_answer_prompt_carries_persona_tool_result_and_notes(personas):
    msgs = build_answer_messages(
        "Prime à Lausanne ?",
        personas["buenzli"],
        "fr",
        [("swiss_health_insurance_premiums", {"municipality_code": 5586}, '{"minimum_premium": "531.7"}')],
        ["Assumed an adult aged 30"],
    )
    assert "Bünzli" in _system(msgs)
    assert "Never invent figures" in _system(msgs)
    user = msgs[-1]["content"]
    assert "531.7" in user and "Assumed an adult aged 30" in user and "5586" in user


def test_answer_prompt_keeps_only_recent_history(personas):
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    msgs = build_answer_messages("q", personas["newcomer"], "en", [], [], history)
    assert [m["content"] for m in msgs[1:-1]] == ["m6", "m7", "m8", "m9"]


def test_answer_prompt_numbers_several_lookups(personas):
    lookups = [
        ("swiss_health_insurance_premiums", {"municipality_code": 5586}, "531.7"),
        ("swiss_health_insurance_premiums", {"municipality_code": 6621}, "476.0"),
    ]
    user = build_answer_messages("Lausanne vs Genève?", personas["newcomer"], "fr", lookups, [])[-1]["content"]
    assert "TOOL RESULT 1 of 2" in user and "TOOL RESULT 2 of 2" in user
    assert "531.7" in user and "476.0" in user and "6621" in user


def test_route_prompt_lists_preferred_tools_first(personas):
    system = _system(build_route_messages("q", personas["buenzli"], FakeMCP().tools))
    preferred, others = system.split("Other tools:")
    assert "swiss_school_holidays" in preferred
    assert "swiss_federal_political_rights" in others


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("What is the minimum health insurance premium in Lausanne for an adult?", "en"),
        ("Quelle est la prime minimale d'assurance maladie à Lausanne pour un adulte ?", "fr"),
        ("Wie hoch ist die minimale Krankenkassenprämie in Lausanne für einen Erwachsenen?", "de"),
        ("Qual è il premio minimo dell'assicurazione malattia a Losanna per un adulto?", "it"),
    ],
)
def test_guess_language(text, lang):
    assert guess_language(text) == lang


def test_detect_language_overrides_model_when_stopwords_are_clear():
    from gspaenli.prompts import detect_language

    # Regression: Apertus labelled this English question "fr" because of "Lausanne".
    question = "What is the minimum health insurance premium in Lausanne for an adult?"
    assert detect_language(question, "fr") == "en"


def test_detect_language_uses_model_label_for_short_questions():
    from gspaenli.prompts import detect_language

    assert detect_language("Lausanne?", "fr") == "fr"
    assert detect_language("Grüezi", "de") == "de"
    assert detect_language("Lausanne?", None) == "en"


def test_italian_answer_with_la_and_le_is_italian():
    # Regression: the eval counted this Italian answer as French because of "La" and "Le".
    text = (
        "La cittadinanza svizzera può votare alle elezioni del Consiglio nazionale a partire dai 18 anni. "
        "Le elezioni del Consiglio degli Stati seguono le norme cantonali."
    )
    assert guess_language(text) == "it"


def test_german_answer_with_des_is_german():
    # Regression: the eval counted this German answer as French ("des").
    text = (
        "Ein fakultatives Referendum erfordert 50 000 gültige Unterschriften innerhalb von 100 Tagen seit "
        "Publikation des Erlasses im Bundesblatt. Eine Volksinitiative braucht 100 000 gültige Unterschriften."
    )
    assert guess_language(text) == "de"
