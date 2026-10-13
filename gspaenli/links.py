"""Keep only links that come from the looked-up data.

Small models add plausible URLs ("www.bern.ch/de/abfall") that are not in any result. Every
link left in an answer must appear in the tool output it was based on.
"""

from __future__ import annotations

import re

MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
BARE_URL = re.compile(r"(?:https?://|www\.)[^\s)\]>\"']+")
TRAILING = ".,;:!?"


def _known(url: str, data: str) -> bool:
    core = url.rstrip(TRAILING)
    bare = re.sub(r"^https?://", "", core)
    return core in data or bare in data


def strip_unsourced_links(answer: str, data: str) -> tuple[str, list[str]]:
    """(answer without links that are not in data, removed links)."""
    removed: list[str] = []

    def markdown(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        if _known(url, data):
            return match.group(0)
        removed.append(url)
        return label if not BARE_URL.fullmatch(label) else ""

    text = MARKDOWN_LINK.sub(markdown, answer)

    def bare(match: re.Match[str]) -> str:
        url = match.group(0)
        if _known(url, data):
            return url
        removed.append(url.rstrip(TRAILING))
        return url[len(url.rstrip(TRAILING)) :]  # keep trailing punctuation

    text = BARE_URL.sub(bare, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip(), removed
