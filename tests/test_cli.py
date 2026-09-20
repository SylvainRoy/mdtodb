from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mdtodb.cli as cli
from mdtodb import MANIFEST_NAME
from mdtodb.cli import app

from .conftest import FakeEmbedding
from .test_sync import make_tree

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
