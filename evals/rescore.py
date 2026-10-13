"""Recompute the automatic scores of saved results after a change to the language check.

The answers are not regenerated; only answer_lang, lang_ok and the summary are recomputed.
    uv run python evals/rescore.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.run import answer_language  # noqa: E402

for path in sorted((Path(__file__).parent / "results").glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    for r in rows:
        r["answer_lang"] = answer_language(r["answer"]) if r["answer"] else None
        r["lang_ok"] = r["answer_lang"] == r["expected_lang"]
    data["summary"]["language_accuracy"] = sum(r["lang_ok"] for r in rows) / len(rows)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{path.name}: language {data['summary']['language_accuracy']:.0%}")
