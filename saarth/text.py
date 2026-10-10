"""Small text utilities shared across modules."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an the is are was were be been of to in on for and or with by at from as that this it its "
    "what which who whom how why when where do does did can could should would will shall may "
    "i you he she we they me my your our their about into than then there these those not no".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def sentences(text: str) -> list[str]:
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [s.strip() for s in _SENT.split(body.replace("\n", " ")) if len(s.strip()) > 3]
