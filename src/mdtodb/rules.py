"""Rule-based keyword extraction from a TOML config file.

Each rule adds keywords to a document when its regexes match the document's
relative POSIX path and/or its Markdown content. The file's sha256 is stored
in each item's metadata so a change of rules triggers a metadata-only
refresh (no re-embedding) on the next sync.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Pattern

from .metadata import normalize_keyword

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DEFAULT_RULES_PATH = Path("~/.config/mdtodb/rules.toml")

_OPTION_KEYS = {"ignore_case"}
_RULE_KEYS = {"keywords", "path", "content", "ignore_case"}
_TOP_KEYS = {"options", "rule"}


class RulesError(ValueError):
    """Invalid keyword rules file."""


@dataclass(frozen=True)
class Rule:
    """One ``[[rule]]`` entry; matches when every set regex matches."""

    keywords: tuple[str, ...]
    path: Pattern[str] | None = None
    content: Pattern[str] | None = None

    def matches(self, rel_path: str, markdown: str) -> bool:
        if self.path is not None and not self.path.search(rel_path):
            return False
        if self.content is not None and not self.content.search(markdown):
            return False
        return True


@dataclass(frozen=True)
class KeywordRules:
    rules: tuple[Rule, ...]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "KeywordRules":
        raw = Path(path).read_bytes()
        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise RulesError(f"{path}: {exc}") from exc
        return cls.from_dict(data, sha256=hashlib.sha256(raw).hexdigest())

    @classmethod
    def from_dict(cls, data: dict, *, sha256: str | None = None) -> "KeywordRules":
        if sha256 is None:
            sha256 = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode("utf-8")
            ).hexdigest()
        if not isinstance(data, dict):
            raise RulesError("rules file must be a TOML table")
        unknown = set(data) - _TOP_KEYS
        if unknown:
            raise RulesError(f"unknown top-level key(s): {sorted(unknown)}")

        options = data.get("options", {})
        if not isinstance(options, dict):
            raise RulesError("'options' must be a table")
        unknown = set(options) - _OPTION_KEYS
        if unknown:
            raise RulesError(f"unknown key(s) in [options]: {sorted(unknown)}")
        default_ignore_case = options.get("ignore_case", True)
        if not isinstance(default_ignore_case, bool):
            raise RulesError("'options.ignore_case' must be a boolean")

        entries = data.get("rule", [])
        if not isinstance(entries, list):
            raise RulesError("'rule' must be an array of tables")
        rules: list[Rule] = []
        for index, entry in enumerate(entries, 1):
            rules.append(_parse_rule(entry, index, default_ignore_case))
        return cls(rules=tuple(rules), sha256=sha256)

    def keywords_for(self, rel_path: str, markdown: str) -> list[str]:
        """Ordered, deduplicated keywords from every matching rule."""
        out: list[str] = []
        seen: set[str] = set()
        for rule in self.rules:
            if rule.matches(rel_path, markdown):
                for kw in rule.keywords:
                    if kw not in seen:
                        seen.add(kw)
                        out.append(kw)
        return out


def _compile(pattern: object, field: str, index: int, ignore_case: bool) -> Pattern[str]:
    if not isinstance(pattern, str):
        raise RulesError(f"rule {index}: '{field}' must be a string")
    try:
        return re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise RulesError(f"rule {index}: invalid regex in '{field}': {exc}") from exc


def _parse_rule(entry: object, index: int, default_ignore_case: bool) -> Rule:
    if not isinstance(entry, dict):
        raise RulesError(f"rule {index}: must be a table")
    unknown = set(entry) - _RULE_KEYS
    if unknown:
        raise RulesError(f"rule {index}: unknown key(s): {sorted(unknown)}")

    keywords = entry.get("keywords")
    if keywords is None:
        raise RulesError(f"rule {index}: 'keywords' is required")
    if not isinstance(keywords, list) or not keywords:
        raise RulesError(f"rule {index}: 'keywords' must be a non-empty list")
    for kw in keywords:
        if not isinstance(kw, str):
            raise RulesError(f"rule {index}: 'keywords' entries must be strings")

    ignore_case = entry.get("ignore_case", default_ignore_case)
    if not isinstance(ignore_case, bool):
        raise RulesError(f"rule {index}: 'ignore_case' must be a boolean")

    path = entry.get("path")
    content = entry.get("content")
    if path is None and content is None:
        raise RulesError(f"rule {index}: needs at least one of 'path' or 'content'")

    return Rule(
        keywords=tuple(normalize_keyword(kw) for kw in keywords),
        path=_compile(path, "path", index, ignore_case) if path is not None else None,
        content=_compile(content, "content", index, ignore_case) if content is not None else None,
    )


def find_rules_file(explicit: str | Path | None) -> Path | None:
    """Resolve the rules file: explicit arg, then $MDTODB_RULES, then the default path."""
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"rules file not found: {path}")
        return path
    env = os.environ.get("MDTODB_RULES")
    if env:
        path = Path(env)
        if not path.exists():
            raise FileNotFoundError(f"rules file not found: {path}")
        return path
    default = DEFAULT_RULES_PATH.expanduser()
    return default if default.exists() else None


def load_rules(explicit: str | Path | None) -> KeywordRules | None:
    path = find_rules_file(explicit)
    return KeywordRules.load(path) if path is not None else None
