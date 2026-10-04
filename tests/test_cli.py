from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mdtodb.cli as cli
import mdtodb.rules as rules_module
from mdtodb.cli import app

from .conftest import FakeEmbedding
from .test_sync import make_tree, move_entry

runner = CliRunner()

FILES = {"personnes/Estelle/papiers/contrat.pdf": "# Contrat\n\nHello world."}


@pytest.fixture(autouse=True)
def fake_embedding(monkeypatch):
    monkeypatch.setattr(cli, "_embedding", lambda name, model, api_key: (FakeEmbedding(), "fake"))


@pytest.fixture
def md_dir(tmp_path: Path) -> Path:
    return make_tree(tmp_path, FILES)


def test_keywords():
    result = runner.invoke(app, ["keywords", "personnes/Estelle/papiers/Allemagne/ReleveIntegral_ROY_ESTELLE_2014_26-01-2021.pdf"])
    assert result.exit_code == 0
    meta = json.loads(result.stdout)
    assert meta["person"] == "Estelle"
    assert meta["keywords"][:5] == ["personnes", "estelle", "papiers", "allemagne", "releveintegral"]
    assert "2021" in meta["keywords"]
    assert "engine" not in meta


def test_keywords_stopword():
    result = runner.invoke(app, ["keywords", "Le contrat maison.pdf", "-w", "maison"])
    assert json.loads(result.stdout)["keywords"] == ["contrat"]


def test_sync_dry_run(md_dir: Path, tmp_path: Path):
    chroma = tmp_path / "chroma"
    result = runner.invoke(app, ["sync", str(md_dir), str(chroma), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "[           new] personnes/Estelle/papiers/contrat.pdf" in result.stdout
    assert "1 to index" in result.stderr


def test_sync_then_list(md_dir: Path, tmp_path: Path):
    chroma = tmp_path / "chroma"
    result = runner.invoke(app, ["sync", str(md_dir), str(chroma)])
    assert result.exit_code == 0, result.output
    assert "indexed 1" in result.stderr
    result = runner.invoke(app, ["list", str(md_dir), str(chroma), "-v"])
    assert result.exit_code == 0
    assert "up-to-date" in result.stdout
    assert "0 to index" in result.stderr


def test_query(md_dir: Path, tmp_path: Path):
    chroma = tmp_path / "chroma"
    assert runner.invoke(app, ["sync", str(md_dir), str(chroma)]).exit_code == 0
    result = runner.invoke(app, ["query", str(chroma), "contrat"])
    assert result.exit_code == 0, result.output
    assert "personnes/Estelle/papiers/contrat.pdf" in result.stdout
    assert "person: Estelle" in result.stdout


def test_query_filters(md_dir: Path, tmp_path: Path):
    chroma = tmp_path / "chroma"
    assert runner.invoke(app, ["sync", str(md_dir), str(chroma)]).exit_code == 0
    hit = runner.invoke(app, ["query", str(chroma), "x", "--person", "Estelle", "-k", "contrat", "--filetype", "pdf"])
    assert "contrat.pdf" in hit.stdout
    miss = runner.invoke(app, ["query", str(chroma), "x", "--person", "Nobody"])
    assert "contrat.pdf" not in miss.stdout


def test_neither_nor_both_targets(md_dir: Path, tmp_path: Path):
    result = runner.invoke(app, ["sync", str(md_dir)])
    assert result.exit_code == 2
    result = runner.invoke(app, ["sync", str(md_dir), str(tmp_path / "c"), "--host", "localhost"])
    assert result.exit_code == 2


def test_missing_manifest(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["sync", str(empty), str(tmp_path / "c")])
    assert result.exit_code == 2
    assert "manifest" in result.stderr


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "mdtodb" in result.stdout


# -- keyword rules -----------------------------------------------------------

RULES_TOML = '[[rule]]\nkeywords=["tagged"]\npath="contrat"\n'


@pytest.fixture(autouse=True)
def no_user_rules_file(tmp_path: Path, monkeypatch):
    """Never pick up a real ~/.config/mdtodb/rules.toml or MDTODB_RULES."""
    monkeypatch.delenv("MDTODB_RULES", raising=False)
    monkeypatch.setattr(rules_module, "DEFAULT_RULES_PATH", tmp_path / "home" / ".config" / "mdtodb" / "rules.toml")


def write_rules(tmp_path: Path, text: str = RULES_TOML) -> Path:
    p = tmp_path / "rules.toml"
    p.write_text(text, "utf-8")
    return p


def test_keywords_with_rules(tmp_path: Path):
    rules = write_rules(tmp_path)
    md = tmp_path / "doc.md"
    md.write_text("# Hello")
    result = runner.invoke(
        app, ["keywords", "docs/contrat.pdf", "--rules", str(rules), "--markdown", str(md)]
    )
    assert result.exit_code == 0, result.output
    meta = json.loads(result.stdout)
    assert "tagged" in meta["keywords"]
    assert meta["rules_sha256"]


def test_keywords_invalid_rules(tmp_path: Path):
    bad = write_rules(tmp_path, "not = [toml\n")
    result = runner.invoke(app, ["keywords", "x.pdf", "--rules", str(bad)])
    assert result.exit_code == 2
    result = runner.invoke(app, ["keywords", "x.pdf", "--rules", str(tmp_path / "missing.toml")])
    assert result.exit_code == 2


def test_retag_flow(md_dir: Path, tmp_path: Path):
    chroma = tmp_path / "chroma"
    rules = write_rules(tmp_path)
    result = runner.invoke(app, ["sync", str(md_dir), str(chroma), "--rules", str(rules)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["retag", str(md_dir), str(chroma), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "[         retag] personnes/Estelle/papiers/contrat.pdf" in result.stdout
    result = runner.invoke(app, ["retag", str(md_dir), str(chroma)])
    assert result.exit_code == 0, result.output
    assert "indexed 1" in result.stderr


def test_move_flow(md_dir: Path, tmp_path: Path, monkeypatch):
    chroma = tmp_path / "chroma"
    assert runner.invoke(app, ["sync", str(md_dir), str(chroma)]).exit_code == 0
    old = "personnes/Estelle/papiers/contrat.pdf"
    new = "archives/renamed.pdf"
    move_entry(md_dir, old, new)
    for command in (["list"], ["sync", "--dry-run"]):
        result = runner.invoke(app, command + [str(md_dir), str(chroma)])
        assert result.exit_code == 0, result.output
        assert f"[         moved] {old} -> {new}" in result.stdout
        assert "0 orphan item(s)" in result.stderr
        assert "[           new]" not in result.stdout

    with monkeypatch.context() as patch:
        patch.setattr(FakeEmbedding, "__call__", lambda self, input: pytest.fail("move re-embedded"))
        result = runner.invoke(app, ["sync", str(md_dir), str(chroma)])
    assert result.exit_code == 0, result.output
    assert f"{old} -> {new} (moved)" in result.stderr
    assert "indexed 0, moved 1, failed 0, pruned 0" in result.stderr
    result = runner.invoke(app, ["list", str(md_dir), str(chroma), "-v"])
    assert result.exit_code == 0, result.output
    assert f"[    up-to-date] {new}" in result.stdout
    assert "0 to index" in result.stderr and "0 orphan item(s)" in result.stderr
    result = runner.invoke(app, ["query", str(chroma), "contrat"])
    assert result.exit_code == 0, result.output
    assert new in result.stdout and old not in result.stdout
    assert "person: Estelle" not in result.stdout


def test_failed_item_reports_action(md_dir: Path, tmp_path: Path, monkeypatch):
    chroma = tmp_path / "chroma"
    rel = "personnes/Estelle/papiers/contrat.pdf"

    def boom(self, input):
        raise RuntimeError("embedding down")

    with monkeypatch.context() as patch:
        patch.setattr(FakeEmbedding, "__call__", boom)
        result = runner.invoke(app, ["sync", str(md_dir), str(chroma)])
    assert result.exit_code == 1
    assert f"[1/1] {rel} (new) FAILED: " in result.stderr and "embedding down" in result.stderr

    assert runner.invoke(app, ["sync", str(md_dir), str(chroma)]).exit_code == 0
    new = "archives/renamed.pdf"
    move_entry(md_dir, rel, new)
    (md_dir / rel).with_suffix(".md").unlink(missing_ok=True)

    def failing_move(self, item, markdown, meta):
        raise RuntimeError("move down")

    with monkeypatch.context() as patch:
        patch.setattr(cli.Indexer, "_move", failing_move)
        result = runner.invoke(app, ["sync", str(md_dir), str(chroma)])
    assert result.exit_code == 1
    assert f"[1/1] {rel} -> {new} (moved) FAILED: move down" in result.stderr
