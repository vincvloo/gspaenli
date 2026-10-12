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
