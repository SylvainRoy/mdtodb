from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

import mdtodb.rules as rules_module
from mdtodb.metadata import metadata_for
from mdtodb.rules import KeywordRules, RulesError, find_rules_file, load_rules

VALID = """\
[options]
ignore_case = true

[[rule]]
keywords = ["impots", "fiscal"]
path = 'imp[oô]ts|taxe'

[[rule]]
keywords = ["facture"]
content = 'avis\\s+d.imp[oô]t'
"""


def write(tmp_path: Path, text: str, name: str = "rules.toml") -> Path:
    p = tmp_path / name
    p.write_text(text, "utf-8")
    return p


def test_load_valid_file(tmp_path: Path):
    path = write(tmp_path, VALID)
    rules = KeywordRules.load(path)
    assert len(rules.rules) == 2
    assert rules.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rules.rules[0].path.search("docs/IMPOTS 2024.pdf")


def test_load_literal_string_regex(tmp_path: Path):
    # A plain literal (no regex metachars) is a valid pattern.
    rules = KeywordRules.load(write(tmp_path, '[[rule]]\nkeywords=["x"]\npath="contrat"\n'))
    assert rules.keywords_for("a/contrat.pdf", "") == ["x"]
    assert rules.keywords_for("a/autre.pdf", "") == []


def test_from_dict_sha(tmp_path: Path):
    data = {"rule": [{"keywords": ["a"], "path": "x"}]}
    rules = KeywordRules.from_dict(data)
    import json
    assert rules.sha256 == hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize(
    "toml, match",
    [
        ('[[rule]]\nkeywords=["a"]\npath="("\n', "rule 1: invalid regex in 'path'"),
        ('[[rule]]\nkeywords=["a"]\ncontent="["\n', "rule 1"),
        ('[[rule]]\nkeywords=["a"]\npath="ok"\n[[rule]]\nkeywords=["b"]\ncontent="["\n', "rule 2"),
        ('[[rule]]\nkeywords=[]\npath="x"\n', "rule 1: 'keywords' must be a non-empty list"),
        ('[[rule]]\npath="x"\n', "rule 1: 'keywords' is required"),
        ('[[rule]]\nkeywords=["a", 3]\npath="x"\n', "rule 1: 'keywords' entries must be strings"),
        ('[[rule]]\nkeywords=["a"]\n', "rule 1: needs at least one of 'path' or 'content'"),
        ('[[rule]]\nkeywords=["a"]\npath="x"\nunknown=1\n', "rule 1: unknown key"),
        ('[bogus]\nx=1\n', "unknown top-level key"),
        ('[options]\nunknown=1\n', "unknown key(s) in [options]"),
    ],
)
def test_validation_errors(tmp_path: Path, toml: str, match: str):
    with pytest.raises(RulesError, match=re.escape(match)):
        KeywordRules.load(write(tmp_path, toml))


def test_bad_toml(tmp_path: Path):
    with pytest.raises(RulesError):
        KeywordRules.load(write(tmp_path, "not = [toml\n"))


def test_path_only_match():
    rules = KeywordRules.from_dict({"rule": [{"keywords": ["tag"], "path": "contrat"}]})
    assert rules.keywords_for("p/contrat.pdf", "any text") == ["tag"]
    assert rules.keywords_for("p/autre.pdf", "contrat inside content") == []


def test_content_only_match():
    rules = KeywordRules.from_dict({"rule": [{"keywords": ["tag"], "content": "hello"}]})
    assert rules.keywords_for("any/path.pdf", "say hello world") == ["tag"]
    assert rules.keywords_for("any/hello.pdf", "no match") == []


def test_path_and_content_are_anded():
    rules = KeywordRules.from_dict(
        {"rule": [{"keywords": ["tag"], "path": "contrat", "content": "signe"}]}
    )
    assert rules.keywords_for("p/contrat.pdf", "pas signe ici") == ["tag"]
    assert rules.keywords_for("p/contrat.pdf", "nothing") == []
    assert rules.keywords_for("p/autre.pdf", "signe") == []


def test_ignore_case_default_true():
    rules = KeywordRules.from_dict({"rule": [{"keywords": ["t"], "path": "impots"}]})
    assert rules.keywords_for("IMPOTS.pdf", "") == ["t"]


def test_ignore_case_per_rule_false():
    rules = KeywordRules.from_dict(
        {
            "options": {"ignore_case": True},
            "rule": [
                {"keywords": ["sensitive"], "path": "Impots", "ignore_case": False},
                {"keywords": ["default"], "path": "TAXE"},
            ],
        }
    )
    assert rules.keywords_for("impots.pdf", "") == []
    assert rules.keywords_for("Impots.pdf", "") == ["sensitive"]
    assert rules.keywords_for("taxe.pdf", "") == ["default"]  # [options] default still true


def test_keyword_normalization():
    rules = KeywordRules.from_dict({"rule": [{"keywords": ["Impôts", "Fiscalité"], "path": "x"}]})
    assert rules.keywords_for("x", "") == ["impots", "fiscalite"]


def test_dedup_across_rules():
    rules = KeywordRules.from_dict(
        {
            "rule": [
                {"keywords": ["a", "b"], "path": "x"},
                {"keywords": ["b", "c"], "content": "y"},
            ]
        }
    )
    assert rules.keywords_for("x", "y") == ["a", "b", "c"]


def test_find_rules_file_explicit_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_rules_file(tmp_path / "nope.toml")


def test_find_rules_file_order(tmp_path: Path, monkeypatch):
    explicit = write(tmp_path, VALID, "explicit.toml")
    env = write(tmp_path, VALID, "env.toml")
    default = tmp_path / "default" / "rules.toml"
    default.parent.mkdir()
    default.write_text(VALID)
    monkeypatch.setattr(rules_module, "DEFAULT_RULES_PATH", default)
    monkeypatch.setenv("MDTODB_RULES", str(env))
    assert find_rules_file(explicit) == explicit
    assert find_rules_file(None) == env
    monkeypatch.delenv("MDTODB_RULES")
    assert find_rules_file(None) == default
    default.unlink()
    assert find_rules_file(None) is None


def test_load_rules_none(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MDTODB_RULES", raising=False)
    monkeypatch.setattr(rules_module, "DEFAULT_RULES_PATH", tmp_path / "missing.toml")
    assert load_rules(None) is None


def test_metadata_for_with_rules():
    rules = KeywordRules.from_dict(
        {"rule": [{"keywords": ["fiscal", "impots"], "path": "avis"}]}
    )
    meta = metadata_for("impots/avis-2024.pdf", rules=rules, text="")
    # path keywords first, rule keywords appended, "impots" not duplicated
    assert meta["keywords"] == ["impots", "avis", "2024", "fiscal"]
    assert meta["rules_sha256"] == rules.sha256


def test_metadata_for_rule_no_match_still_sets_sha():
    rules = KeywordRules.from_dict({"rule": [{"keywords": ["zzz"], "path": "nomatch"}]})
    meta = metadata_for("docs/avis.pdf", rules=rules, text="")
    assert meta["rules_sha256"] == rules.sha256
    assert "zzz" not in meta["keywords"]


def test_metadata_for_without_rules():
    meta = metadata_for("docs/avis.pdf")
    assert "rules_sha256" not in meta
