from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdtodb import MANIFEST_NAME, Indexer, Manifest, Reason
from mdtodb.sync import md_sha256


def make_tree(tmp_path: Path, files: dict[str, str], **entry_overrides) -> Path:
    """Create a pdftomd-style output dir: manifest + one .md per rel source path."""
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    manifest_files = {}
    for rel, text in files.items():
        out = rel.rsplit(".", 1)[0] + ".md"
        (md_dir / out).parent.mkdir(parents=True, exist_ok=True)
        (md_dir / out).write_text(text, "utf-8")
        entry = {"fingerprint": f"fp:{rel}", "engine": "fake", "output": out}
        entry.update(entry_overrides.get(rel, {}))
        manifest_files[rel] = entry
    (md_dir / MANIFEST_NAME).write_text(json.dumps({"version": 2, "files": manifest_files}), "utf-8")
    return md_dir


FILES = {
    "personnes/Estelle/papiers/contrat.pdf": "# Contrat\n\nHello world.",
    "docs/notes.txt": "Some notes.",
}


@pytest.fixture
def md_dir(tmp_path: Path) -> Path:
    return make_tree(tmp_path, FILES)


@pytest.fixture
def indexer(md_dir: Path, collection) -> Indexer:
    return Indexer(md_dir, collection, embedding_name="fake")


def test_plan_all_new(indexer: Indexer):
    plan = indexer.plan()
    assert {i.rel_path for i in plan.to_index} == set(FILES)
    assert all(i.reason == Reason.NEW for i in plan.to_index)
    assert not plan.up_to_date and not plan.orphans and not plan.errors


def test_execute_and_metadata(indexer: Indexer, collection):
    result = indexer.execute(indexer.plan())
    assert sorted(result.indexed) == sorted(FILES)
    got = collection.get(include=["metadatas", "documents"])
    state = dict(zip(got["ids"], got["metadatas"]))
    meta = state["personnes/Estelle/papiers/contrat.pdf"]
    assert meta["file"] == "personnes/Estelle/papiers/contrat.pdf"
    assert meta["markdown"] == "personnes/Estelle/papiers/contrat.md"
    assert meta["person"] == "Estelle"
    assert meta["filetype"] == "pdf"
    assert "contrat" in meta["keywords"] and "estelle" in meta["keywords"]
    assert meta["engine"] == "fake"
    assert meta["fingerprint"] == "fp:personnes/Estelle/papiers/contrat.pdf"
    assert meta["md_sha256"] == md_sha256(FILES["personnes/Estelle/papiers/contrat.pdf"])
    assert meta["embedding"] == "fake"
    assert meta["indexed_at"] > 0
    assert state["docs/notes.txt"]["filetype"] == "txt"


def test_second_plan_up_to_date(indexer: Indexer):
    indexer.execute(indexer.plan())
    plan = indexer.plan()
    assert not plan.to_index
    assert sorted(plan.up_to_date) == sorted(FILES)


def test_fingerprint_change(indexer: Indexer, md_dir: Path):
    indexer.execute(indexer.plan())
    data = json.loads((md_dir / MANIFEST_NAME).read_text())
    data["files"]["docs/notes.txt"]["fingerprint"] = "different"
    (md_dir / MANIFEST_NAME).write_text(json.dumps(data))
    indexer.reload()
    plan = indexer.plan()
    assert [(i.rel_path, i.reason) for i in plan.to_index] == [("docs/notes.txt", Reason.CHANGED)]


def test_markdown_change(indexer: Indexer, md_dir: Path):
    indexer.execute(indexer.plan())
    (md_dir / "docs/notes.md").write_text("edited content", "utf-8")
    plan = indexer.plan()
    assert [(i.rel_path, i.reason) for i in plan.to_index] == [("docs/notes.txt", Reason.MARKDOWN_CHANGED)]


def test_embedding_change(indexer: Indexer, collection):
    indexer.execute(indexer.plan())
    other = Indexer(md_dir_fixture(indexer), collection, embedding_name="other")
    plan = other.plan()
    assert all(i.reason == Reason.EMBEDDING_CHANGED for i in plan.to_index)


def md_dir_fixture(indexer: Indexer) -> Path:
    return indexer.md_dir


def test_orphan_and_prune(indexer: Indexer, md_dir: Path, collection):
    indexer.execute(indexer.plan())
    data = json.loads((md_dir / MANIFEST_NAME).read_text())
    del data["files"]["docs/notes.txt"]
    (md_dir / MANIFEST_NAME).write_text(json.dumps(data))
    indexer.reload()
    plan = indexer.plan()
    assert plan.orphans == ["docs/notes.txt"]
    indexer.execute(plan)  # no prune: orphan survives
    assert "docs/notes.txt" in collection.get()["ids"]
    result = indexer.execute(indexer.plan(), prune=True)
    assert result.pruned == ["docs/notes.txt"]
    assert "docs/notes.txt" not in collection.get()["ids"]


def test_select_forces(indexer: Indexer):
    indexer.execute(indexer.plan())
    plan = indexer.plan(select=["docs/notes.txt"])
    assert [(i.rel_path, i.reason) for i in plan.to_index] == [("docs/notes.txt", Reason.FORCED)]
    assert not plan.up_to_date  # restricted to the selection


def test_select_unknown(indexer: Indexer):
    with pytest.raises(FileNotFoundError):
        indexer.plan(select=["nope.pdf"])


def test_force(indexer: Indexer):
    indexer.execute(indexer.plan())
    plan = indexer.plan(force=True)
    assert all(i.reason == Reason.FORCED for i in plan.to_index)


def test_missing_markdown_is_error(indexer: Indexer, md_dir: Path):
    (md_dir / "docs/notes.md").unlink()
    plan = indexer.plan()
    assert "docs/notes.txt" in plan.errors
    assert [i.rel_path for i in plan.to_index] == ["personnes/Estelle/papiers/contrat.pdf"]
    result = indexer.execute(plan)
    assert "docs/notes.txt" in result.failed


def test_empty_markdown_is_error(tmp_path: Path, collection):
    md_dir = make_tree(tmp_path, {"a/b.pdf": ""})
    indexer = Indexer(md_dir, collection, embedding_name="fake")
    result = indexer.execute(indexer.plan())
    assert result.failed["a/b.pdf"] == "empty markdown"
    assert not collection.get()["ids"]


def test_diskless(collection):
    manifest = Manifest.from_entries({"x/y.pdf": {"fingerprint": "f", "engine": "e", "output": "x/y.md"}})
    indexer = Indexer(
        None,
        collection,
        manifest=manifest,
        embedding_name="fake",
        read_markdown=lambda rel: f"text of {rel}",
    )
    result = indexer.execute(indexer.plan())
    assert result.indexed == ["x/y.pdf"]
    meta = collection.get(ids=["x/y.pdf"], include=["metadatas"])["metadatas"][0]
    assert meta["markdown"] == "x/y.md"


def test_index_document_and_remove(collection):
    indexer = Indexer(
        None,
        collection,
        manifest=Manifest(),
        embedding_name="fake",
        read_markdown=lambda rel: "",
    )
    meta = indexer.index_document("personnes/E/doc.pdf", "# Doc", engine="e", fingerprint="f")
    assert meta["person"] == "E"
    assert meta["fingerprint"] == "f"
    assert collection.get()["ids"] == ["personnes/E/doc.pdf"]
    indexer.remove("personnes/E/doc.pdf")
    assert collection.get()["ids"] == []


def test_missing_manifest(tmp_path: Path, collection):
    with pytest.raises(FileNotFoundError, match="manifest"):
        Indexer(tmp_path / "empty", collection)


def test_sync_shorthand(indexer: Indexer):
    result = indexer.sync()
    assert sorted(result.indexed) == sorted(FILES)
    assert not indexer.plan().to_index
