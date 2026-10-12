import yaml

from gspaenli.config import ROOT
from gspaenli.mcp_client import ALLOWED_TOOLS


def test_eval_questions_cover_languages_and_cross_language():
    questions = yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text(encoding="utf-8"))
    assert len(questions) >= 10
    assert {q["lang"] for q in questions} >= {"en", "fr", "de", "it"}
    assert any(q.get("source_lang") and q["source_lang"] != q["lang"] for q in questions)
    assert all(q["expected_tool"] in ALLOWED_TOOLS for q in questions)
    assert len({q["id"] for q in questions}) == len(questions)
