import yaml

from gspaenli.config import ROOT
from gspaenli.mcp_client import ALLOWED_TOOLS


def test_eval_questions_cover_languages_and_cross_language():
    questions = yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text(encoding="utf-8"))
    assert len(questions) >= 10
    assert {q["lang"] for q in questions} >= {"en", "fr", "de", "it"}
    assert any(q.get("source_lang") and q["source_lang"] != q["lang"] for q in questions)
    assert all(q["expected_tool"] in ALLOWED_TOOLS for q in questions)
    multi = [q for q in questions if q.get("required_tools")]
    assert len(multi) >= 4 and {q["lang"] for q in multi} == {"en", "fr", "de", "it"}
    assert all(set(q["required_tools"]) <= ALLOWED_TOOLS and len(q["required_tools"]) >= 2 for q in multi)
    assert len({q["id"] for q in questions}) == len(questions)


def test_answer_language_ignores_urls_and_official_terms():
    import importlib.util

    spec = importlib.util.spec_from_file_location("evals_run", ROOT / "evals" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    english = (
        "A federal popular initiative (eidgenössische Volksinitiative, die Unterschriften der Stimmberechtigten) "
        "needs 100,000 signatures. Source: https://www.bk.admin.ch/de/die-volksinitiative-und-das-referendum"
    )
    assert run.answer_language(english) == "en"
    assert run.answer_language("Les vacances d'automne sont du 10 au 25 octobre.") == "fr"
