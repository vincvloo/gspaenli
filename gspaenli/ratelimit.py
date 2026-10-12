"""In-memory limits so a public URL cannot burn the GPU budget. The UI runs in one container."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime


class RateLimiter:
    def __init__(
        self,
        per_session: int = 20,
        per_day: int = 300,
        min_interval_s: float = 3.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.per_session = per_session
        self.per_day = per_day
        self.min_interval_s = min_interval_s
        self.clock = clock
        self._day = ""
        self._today = 0

    def _roll_day(self) -> None:
        day = datetime.fromtimestamp(self.clock(), UTC).strftime("%Y-%m-%d")
        if day != self._day:
            self._day, self._today = day, 0

    def check(self, session: dict) -> str | None:
        """None if allowed (and counted), otherwise a message for the user."""
        self._roll_day()
        now = self.clock()
        if session.get("turns", 0) >= self.per_session:
            return f"This demo allows {self.per_session} questions per session. Reload the page to start over."
        if now - session.get("last", 0.0) < self.min_interval_s:
            return "One question at a time, please."
        if self._today >= self.per_day:
            return "The demo reached its daily limit. Please come back tomorrow."
        session["turns"] = session.get("turns", 0) + 1
        session["last"] = now
        self._today += 1
        return None
