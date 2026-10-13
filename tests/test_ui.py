from gspaenli.orchestrator import Trace
from gspaenli.ratelimit import RateLimiter
from gspaenli.ui import format_trace


def _trace(**kw):
    base = {"persona": "newcomer", "model_id": "swiss-ai/Apertus-8B-Instruct-2509"}
    return Trace(**{**base, **kw})


def test_trace_shows_tool_args_sources_latency_model():
    md = format_trace(
        _trace(
            tool="swiss_health_insurance_premiums",
            arguments={"age": 30, "municipality_code": 5586, "franchise": 300},
            sources=["https://www.priminfo.admin.ch/de/praemien"],
            notes=["Assumed an adult aged 30"],
            lang="fr",
            latency_s={"route": 1.04, "tool": 0.5, "answer": 4.9, "total": 6.5},
        )
    )
    assert "`swiss_health_insurance_premiums`" in md
    assert '"municipality_code": 5586' in md
    assert "[https://www.priminfo.admin.ch/de/praemien](https://www.priminfo.admin.ch/de/praemien)" in md
    assert "Assumed an adult aged 30" in md
    assert "route 1.0 s, tool 0.5 s, answer 4.9 s, total 6.5 s" in md
    assert "Apertus-8B-Instruct-2509" in md and "reply language:** fr" in md
    assert "—" not in md  # house style: no em-dashes


def test_trace_without_tool_and_with_error_and_fallback():
    assert "none, no lookup" in format_trace(_trace())
    md = format_trace(_trace(tool="search_knowledge", fallback=True, tool_error="timeout"))
    assert "full-text search" in md and "timeout" in md and "none returned" in md


def test_rate_limiter_per_session_interval_and_day():
    now = [1_800_000_000.0]
    limiter = RateLimiter(per_session=2, per_day=3, min_interval_s=5, clock=lambda: now[0])
    a, b = {}, {}
    assert limiter.check(a) is None
    assert "one question at a time" in limiter.check(a).lower()
    now[0] += 10
    assert limiter.check(a) is None
    now[0] += 10
    assert "per session" in limiter.check(a)
    assert limiter.check(b) is None
    now[0] += 10
    assert "daily limit" in limiter.check({})
    now[0] += 86_400
    assert limiter.check({}) is None  # new day resets the global counter


def test_status_pill_states():
    from gspaenli.ui_assets import status_html

    assert "gs-pill-ready" in status_html("ready") and "Model ready" in status_html("ready")
    assert "gs-pill-waking" in status_html("waking")
    assert "asleep" in status_html("asleep")


def test_persona_css_escapes_taglines(personas):
    from gspaenli.ui_assets import persona_css

    css = persona_css(list(personas.values()))
    assert css.count("::after") == len(personas)
    assert '"Rules are rules. But I\'ll help you. This time."' in css


def test_tour_targets_exist_in_ui():
    from pathlib import Path

    from gspaenli.ui_assets import TOUR_STEPS

    ui_source = Path("gspaenli/ui.py").read_text(encoding="utf-8")
    for selector, _, _ in TOUR_STEPS:
        assert f'elem_id="{selector[1:]}"' in ui_source, selector


def test_ui_text_has_no_em_dashes():
    from pathlib import Path

    for path in ["gspaenli/ui.py", "gspaenli/ui_assets.py", *map(str, Path("personas").glob("*.yaml"))]:
        assert "—" not in Path(path).read_text(encoding="utf-8"), path


def test_model_state_never_needs_the_network(monkeypatch):
    import time

    from gspaenli.ui import READY_WINDOW_S, Runtime

    rt = Runtime.__new__(Runtime)  # no LLM client or avatars needed
    rt.waking, rt.last_ok = False, 0.0
    assert rt.model_state() == "asleep"
    rt.last_ok = time.time()
    assert rt.model_state() == "ready"
    rt.last_ok = time.time() - READY_WINDOW_S - 1
    assert rt.model_state() == "asleep"
    rt.waking = True
    assert rt.model_state() == "waking"


def test_ready_pill_carries_its_expiry_for_the_browser():
    from gspaenli.ui_assets import status_html, tour_head

    assert 'data-ready-until="1700000000000"' in status_html("ready", 1_700_000_000_000)
    assert "data-ready-until" not in status_html("asleep", 1_700_000_000_000)
    assert "readyUntil" in tour_head()


def test_ui_has_no_server_timer():
    from pathlib import Path

    assert "gr.Timer" not in Path("gspaenli/ui.py").read_text(encoding="utf-8")
