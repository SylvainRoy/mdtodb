"""Keyword, person and filetype extraction for ChromaDB metadata.

Keywords are derived from the document's path (directories and filename):
each segment is accent-stripped, lowercased and split on non-alphanumeric
characters. Stopwords (a small French + English list, extendable per call),
short tokens and bare numbers are dropped — except 4-digit years in
1900-2099, including years recovered from dates inside a segment.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Iterable

from .manifest import markdown_path_for

if TYPE_CHECKING:
    from .rules import KeywordRules

DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    "le la les l un une des du de d au aux et ou en a a "
    "the an of and or in on for to at by with from".split()
)

_EXT_RE = re.compile(r"\.[0-9A-Za-z]{1,5}$")
_SPLIT_RE = re.compile(r"[^0-9a-z]+")
_YEAR_RE = re.compile(r"^\d{4}$")
# Dates embedded in a path segment; each pattern exposes the 4-digit year.
# Digit lookarounds (not \b) so dates adjacent to _ or letters still match.
_DATE_RES = (
    re.compile(r"(?<!\d)\d{1,2}[-/._]\d{1,2}[-/._](\d{4})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})[-/._]\d{1,2}[-/._]\d{1,2}(?!\d)"),
    re.compile(r"(?<!\d)\d{1,2}[-/._](\d{4})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})[-/._]\d{1,2}(?!\d)"),
)
_COMPACT_DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\d{4}(?!\d)")


def normalize_keyword(text: str) -> str:
    """NFKD-normalise, strip combining marks (accents), lowercase."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _stopword_set(stopwords: Iterable[str] | None, replace: bool) -> set[str]:
    extra = {normalize_keyword(s) for s in stopwords or ()}
    return extra if replace else set(DEFAULT_STOPWORDS) | extra


def _is_year(token: str) -> bool:
    return bool(_YEAR_RE.match(token)) and 1900 <= int(token) <= 2099


def _date_years(segment: str) -> list[str]:
    """Years (1900-2099) found in dates inside one path segment, in order."""
    found: list[tuple[int, str]] = []
    for pattern in _DATE_RES:
        for m in pattern.finditer(segment):
            found.append((m.start(), m.group(1)))
    for m in _COMPACT_DATE_RE.finditer(segment):
        found.append((m.start(), m.group(1)))
    return [y for _, y in sorted(found) if 1900 <= int(y) <= 2099]


def keywords_for(path: str, *, stopwords: Iterable[str] | None = None, replace_stopwords: bool = False) -> list[str]:
    """Ordered, deduplicated keyword list for a document path."""
    stop = _stopword_set(stopwords, replace_stopwords)
    segments = [s for s in PurePosixPath(path).parts if s]
    if segments:
        segments[-1] = _EXT_RE.sub("", segments[-1])
    out: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        normalized = normalize_keyword(segment)
        tokens = [t for t in _SPLIT_RE.split(normalized) if t]
        tokens += _date_years(normalized)
        for token in tokens:
            if len(token) < 2 or token in stop:
                continue
            if token.isdigit() and not _is_year(token):
                continue
            if token not in seen:
                seen.add(token)
                out.append(token)
    return out


def person_for(path: str) -> str | None:
    """Directory name right after a ``personnes`` component, if any.

    ``personnes/Estelle/papiers/x.pdf`` -> ``Estelle``; None when there is no
    ``personnes`` component or it only introduces the file itself.
    """
    parts = PurePosixPath(path).parts
    for i, part in enumerate(parts):
        if part.lower() == "personnes" and i + 1 < len(parts) - 1:
            return parts[i + 1]
    return None


def filetype_for(path: str) -> str:
    """Lowercase extension without the dot; ``""`` when there is none."""
    return PurePosixPath(path).suffix.lstrip(".").lower()


def metadata_for(
    rel_path: str,
    *,
    markdown: str | None = None,
    engine: str | None = None,
    fingerprint: str | None = None,
    md_sha256: str | None = None,
    embedding: str | None = None,
    indexed_at: float | None = None,
    stopwords: Iterable[str] | None = None,
    text: str | None = None,
    rules: KeywordRules | None = None,
) -> dict:
    """Build the Chroma metadata dict for a source document.

    Keys whose value would be None or an empty list/string are omitted —
    Chroma rejects both.
    """
    meta: dict = {"file": rel_path}
    meta["markdown"] = markdown if markdown is not None else markdown_path_for(rel_path)
    filetype = filetype_for(rel_path)
    if filetype:
        meta["filetype"] = filetype
    person = person_for(rel_path)
    if person is not None:
        meta["person"] = person
    keywords = keywords_for(rel_path, stopwords=stopwords)
    if rules is not None:
        seen = set(keywords)
        for kw in rules.keywords_for(rel_path, text or ""):
            if kw not in seen:
                seen.add(kw)
                keywords.append(kw)
        meta["rules_sha256"] = rules.sha256
    if keywords:
        meta["keywords"] = keywords
    for key, value in (
        ("engine", engine),
        ("fingerprint", fingerprint),
        ("md_sha256", md_sha256),
        ("embedding", embedding),
        ("indexed_at", indexed_at),
    ):
        if value is not None:
            meta[key] = value
    return meta
