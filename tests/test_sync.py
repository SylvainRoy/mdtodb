from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdtodb import MANIFEST_NAME, Indexer, Manifest, Reason
from mdtodb.rules import KeywordRules
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


# -- keyword rules -----------------------------------------------------------

RULES = KeywordRules.from_dict({"rule": [{"keywords": ["tagged"], "path": "contrat"}]})


def rules_indexer(md_dir: Path, collection, rules=RULES) -> Indexer:
    return Indexer(md_dir, collection, embedding_name="fake", rules=rules)


def test_rules_indexing(md_dir: Path, collection):
    indexer = rules_indexer(md_dir, collection)
    indexer.execute(indexer.plan())
    state = dict(zip(*[collection.get(include=["metadatas"])[k] for k in ("ids", "metadatas")]))
    meta = state["personnes/Estelle/papiers/contrat.pdf"]
    assert "tagged" in meta["keywords"]
    assert meta["rules_sha256"] == RULES.sha256
    other = state["docs/notes.txt"]
    assert "tagged" not in other["keywords"]
    assert other["rules_sha256"] == RULES.sha256  # sha recorded even without match


def test_rules_second_plan_up_to_date(md_dir: Path, collection):
    indexer = rules_indexer(md_dir, collection)
    indexer.execute(indexer.plan())
    plan = indexer.plan()
    assert not plan.to_index


def test_rules_change_is_metadata_only(md_dir: Path, collection, monkeypatch):
    indexer = rules_indexer(md_dir, collection)
    indexer.execute(indexer.plan())
    other_rules = KeywordRules.from_dict({"rule": [{"keywords": ["newtag"], "path": "contrat"}]})
    other = Indexer(md_dir, collection, embedding_name="fake", rules=other_rules)
    plan = other.plan()
    items = {i.rel_path: i for i in plan.to_index}
    assert items["personnes/Estelle/papiers/contrat.pdf"].reason == Reason.RULES_CHANGED
    assert items["personnes/Estelle/papiers/contrat.pdf"].metadata_only
    assert items["docs/notes.txt"].reason == Reason.RULES_CHANGED

    monkeypatch.setattr(collection, "upsert", lambda **kw: pytest.fail("upsert called"))
    updates = []
    orig_update = collection.update
    monkeypatch.setattr(collection, "update", lambda **kw: (updates.append(kw), orig_update(**kw)))
    before = collection.get(ids=["personnes/Estelle/papiers/contrat.pdf"], include=["metadatas"])["metadatas"][0]
    result = other.execute(plan)
    assert len(result.indexed) == 2
    assert updates and "documents" not in updates[0]
    meta = collection.get(ids=["personnes/Estelle/papiers/contrat.pdf"], include=["metadatas"])["metadatas"][0]
    assert meta["rules_sha256"] == other_rules.sha256
    assert "newtag" in meta["keywords"] and "tagged" not in meta["keywords"]
    assert meta["indexed_at"] >= before["indexed_at"]


def test_rules_removed_clears_metadata(md_dir: Path, collection):
    indexer = rules_indexer(md_dir, collection)
    indexer.execute(indexer.plan())
    plain = Indexer(md_dir, collection, embedding_name="fake")  # no rules
    plan = plain.plan()
    assert all(i.reason == Reason.RULES_CHANGED and i.metadata_only for i in plan.to_index)
    plain.execute(plan)
    meta = collection.get(ids=["personnes/Estelle/papiers/contrat.pdf"], include=["metadatas"])["metadatas"][0]
    assert "rules_sha256" not in meta
    assert "tagged" not in meta["keywords"]


def test_retag(md_dir: Path, collection):
    indexer = rules_indexer(md_dir, collection)
    # unindexed docs are NEW, not RETAG
    plan = indexer.plan(retag=True)
    assert all(i.reason == Reason.NEW and not i.metadata_only for i in plan.to_index)
    indexer.execute(plan)
    plan = indexer.plan(retag=True)
    assert all(i.reason == Reason.RETAG and i.metadata_only for i in plan.to_index)
    assert len(plan.to_index) == len(FILES)
    result = indexer.execute(plan)
    assert len(result.indexed) == len(FILES)


def test_index_document_applies_rules(collection):
    indexer = Indexer(
        None,
        collection,
        manifest=Manifest(),
        embedding_name="fake",
        read_markdown=lambda rel: "",
        rules=KeywordRules.from_dict({"rule": [{"keywords": ["tagged"], "content": "doc"}]}),
    )
    meta = indexer.index_document("a/b.pdf", "# Doc", engine="e", fingerprint="f")
    assert "tagged" in meta["keywords"]
    assert meta["rules_sha256"]


# -- moved documents ---------------------------------------------------------

def move_entry(md_dir: Path, old: str, new: str, *, output: str | None = None) -> None:
    """Apply pdftomd's move: relocate Markdown and rekey the manifest entry."""
    manifest_path = md_dir / MANIFEST_NAME
    data = json.loads(manifest_path.read_text())
    entry = data["files"].pop(old)
    old_output = md_dir / entry["output"]
    entry["output"] = output or str(Path(new).with_suffix(".md"))
    new_output = md_dir / entry["output"]
    new_output.parent.mkdir(parents=True, exist_ok=True)
    old_output.rename(new_output)
    data["files"][new] = entry
    manifest_path.write_text(json.dumps(data))


OLD = "personnes/Estelle/papiers/contrat.pdf"
NEW = "archives/renamed.pdf"


@pytest.mark.parametrize("prune", [False, True])
def test_move_reuses_embedding_and_refreshes_metadata(md_dir: Path, collection, monkeypatch, prune):
    rules = KeywordRules.from_dict({"rule": [
        {"keywords": ["oldtag"], "path": "personnes"},
        {"keywords": ["newtag"], "path": "archives"},
        {"keywords": ["contenttag"], "content": "Hello"},
    ]})
    indexer = rules_indexer(md_dir, collection, rules)
    indexer.sync()
    collection.update(ids=[OLD], metadatas=[{"custom": "keep me"}])
    before = collection.get(ids=[OLD], include=["documents", "embeddings"])
    move_entry(md_dir, OLD, NEW, output="custom/output.md")
    indexer.reload()
    snapshot = {p: p.read_bytes() for p in md_dir.rglob("*") if p.is_file()}

    plan = indexer.plan()
    assert [(i.rel_path, i.reason, i.moved_from, i.metadata_only) for i in plan.to_index] == [
        (NEW, Reason.MOVED, OLD, True)
    ]
    assert plan.to_index[0].markdown == md_dir / "custom/output.md"
    assert plan.orphans == []
    assert sorted(collection.get()["ids"]) == sorted(FILES)  # plan is read-only
    monkeypatch.setattr(collection, "_embed", lambda **kw: pytest.fail("move re-embedded"))
    done, progress = [], []
    result = indexer.execute(
        plan, prune=prune,
        on_done=lambda *args: done.append(args), on_progress=lambda *args: progress.append(args),
    )
    assert result.moved == [NEW]
    assert not result.indexed and not result.failed and not result.pruned
    assert done == progress == [(plan.to_index[0], 1, 1)]
    assert OLD not in collection.get()["ids"]
    after = collection.get(ids=[NEW], include=["documents", "embeddings", "metadatas"])
    assert after["documents"] == before["documents"]
    assert (after["embeddings"] == before["embeddings"]).all()
    meta = after["metadatas"][0]
    assert meta["file"] == NEW and meta["markdown"] == "custom/output.md"
    assert meta["fingerprint"] == f"fp:{OLD}" and meta["engine"] == "fake"
    assert meta["filetype"] == "pdf" and meta["custom"] == "keep me"
    assert "person" not in meta
    assert {"archives", "renamed", "newtag", "contenttag"} <= set(meta["keywords"])
    assert not {"estelle", "contrat", "oldtag"} & set(meta["keywords"])
    assert not indexer.plan().to_index and not indexer.plan().orphans
    assert snapshot == {p: p.read_bytes() for p in md_dir.rglob("*") if p.is_file()}


def test_move_clears_all_optional_metadata(md_dir: Path, collection):
    indexer = rules_indexer(md_dir, collection)
    indexer.sync()
    move_entry(md_dir, OLD, "a")  # no keywords, person or extension at the new path
    plain = Indexer(md_dir, collection, embedding_name="fake")
    assert plain.sync().moved == ["a"]
    meta = collection.get(ids=["a"], include=["metadatas"])["metadatas"][0]
    assert not {"person", "keywords", "filetype", "rules_sha256"} & meta.keys()


def test_move_changes_person(indexer: Indexer, collection):
    indexer.sync()
    new = "personnes/Sylvain/document.txt"
    move_entry(indexer.md_dir, OLD, new)
    indexer.reload()
    indexer.sync()
    meta = collection.get(ids=[new], include=["metadatas"])["metadatas"][0]
    assert meta["person"] == "Sylvain" and meta["filetype"] == "txt"
    assert "sylvain" in meta["keywords"] and "estelle" not in meta["keywords"]


@pytest.mark.parametrize("change", ["markdown", "embedding", "force", "select"])
def test_move_reembeds_when_needed(indexer: Indexer, collection, monkeypatch, change):
    indexer.sync()
    move_entry(indexer.md_dir, OLD, NEW)
    indexer.reload()
    kwargs = {}
    if change == "markdown":
        (indexer.md_dir / "archives/renamed.md").write_text("Changed Markdown")
    elif change == "embedding":
        collection.update(ids=[OLD], metadatas=[{"embedding": "old-model"}])
    elif change == "force":
        kwargs["force"] = True
    else:
        kwargs["select"] = [NEW]
    plan = indexer.plan(**kwargs)
    item = next(i for i in plan.to_index if i.rel_path == NEW)
    assert item.moved_from == OLD and not item.metadata_only
    assert item.reason == {
        "markdown": Reason.MARKDOWN_CHANGED, "embedding": Reason.EMBEDDING_CHANGED,
        "force": Reason.FORCED, "select": Reason.FORCED,
    }[change]
    embedded = []
    original = collection._embed
    monkeypatch.setattr(collection, "_embed", lambda **kw: (embedded.append(kw), original(**kw))[1])
    result = indexer.execute(plan)
    assert result.moved == [NEW] and not result.failed
    assert embedded
    assert OLD not in collection.get()["ids"]
    assert not indexer.plan().to_index


@pytest.mark.parametrize("field", ["fingerprint", "engine"])
def test_changed_source_is_not_a_move(indexer: Indexer, field):
    indexer.sync()
    move_entry(indexer.md_dir, OLD, NEW)
    indexer.reload()
    setattr(indexer.manifest.entries[NEW], field, "changed")
    plan = indexer.plan()
    assert plan.to_index[0].reason == Reason.NEW
    assert plan.to_index[0].moved_from is None
    assert plan.orphans == [OLD]


@pytest.mark.parametrize("field", ["fingerprint", "engine"])
def test_missing_identity_is_not_a_move(indexer: Indexer, collection, field):
    indexer.sync()
    collection.update(ids=[OLD], metadatas=[{field: None}])
    move_entry(indexer.md_dir, OLD, NEW)
    indexer.reload()
    plan = indexer.plan()
    assert plan.to_index[0].reason == Reason.NEW
    assert plan.orphans == [OLD]


def test_copy_is_new_not_moved(indexer: Indexer, collection):
    indexer.sync()
    indexer.manifest.entries[NEW] = indexer.manifest.entries[OLD]
    plan = indexer.plan()
    assert [(i.rel_path, i.reason, i.moved_from) for i in plan.to_index] == [(NEW, Reason.NEW, None)]
    assert not plan.orphans
    result = indexer.execute(plan, prune=True)
    assert result.indexed == [NEW] and not result.moved
    assert OLD in collection.get()["ids"]


@pytest.mark.parametrize("old_count,new_count", [(1, 2), (2, 1), (2, 2)])
def test_duplicate_moves_are_paired_once(collection, monkeypatch, old_count, new_count):
    entry = {"fingerprint": "same", "engine": "fake", "output": ""}
    manifest = Manifest.from_entries({f"old/{i}.pdf": entry for i in range(old_count)})
    indexer = Indexer(None, collection, manifest=manifest, embedding_name="fake", read_markdown=lambda rel: "same")
    indexer.sync()
    indexer.manifest = Manifest.from_entries({f"new/{i}.pdf": entry for i in reversed(range(new_count))})
    plan = indexer.plan()
    moves = {i.rel_path: i.moved_from for i in plan.to_index if i.moved_from is not None}
    assert moves == {f"new/{i}.pdf": f"old/{i}.pdf" for i in range(min(old_count, new_count))}
    if old_count >= new_count:
        monkeypatch.setattr(collection, "_embed", lambda **kw: pytest.fail("move re-embedded"))
    result = indexer.execute(plan, prune=True, batch_size=1)
    assert len(result.moved) == min(old_count, new_count)
    assert len(result.indexed) == max(0, new_count - old_count)
    assert len(result.pruned) == max(0, old_count - new_count)
    assert not result.failed
    assert sorted(collection.get()["ids"]) == [f"new/{i}.pdf" for i in range(new_count)]
    assert not indexer.plan().to_index and not indexer.plan().orphans


def test_select_does_not_prune_unselected_moves(indexer: Indexer, collection):
    indexer.sync()
    move_entry(indexer.md_dir, OLD, NEW)
    move_entry(indexer.md_dir, "docs/notes.txt", "elsewhere/notes.txt")
    indexer.reload()
    plan = indexer.plan(select=[NEW])
    assert not plan.orphans
    result = indexer.execute(plan, prune=True)
    assert result.moved == [NEW]
    assert sorted(collection.get()["ids"]) == [NEW, "docs/notes.txt"]
    assert indexer.sync(prune=True).moved == ["elsewhere/notes.txt"]


@pytest.mark.parametrize("failure", ["plan-read", "execute-read", "empty", "write", "embed", "delete"])
def test_move_failure_preserves_source(indexer: Indexer, collection, monkeypatch, failure):
    indexer.sync()
    move_entry(indexer.md_dir, OLD, NEW)
    indexer.reload()
    if failure == "plan-read":
        (indexer.md_dir / "archives/renamed.md").unlink()
    plan = indexer.plan(force=failure == "embed")
    if failure == "execute-read":
        (indexer.md_dir / "archives/renamed.md").unlink()
    elif failure == "empty":
        (indexer.md_dir / "archives/renamed.md").write_text("")

    def fail(**kwargs):
        raise RuntimeError("simulated failure")

    if failure in ("write", "embed", "delete"):
        monkeypatch.setattr(collection, {"write": "upsert", "embed": "_embed", "delete": "delete"}[failure], fail)
    errors = []
    result = indexer.execute(plan, prune=True, on_error=lambda *args: errors.append(args))
    assert NEW in result.failed
    assert not result.moved and not result.pruned
    assert OLD in collection.get()["ids"]
    assert (NEW in collection.get()["ids"]) == (failure == "delete")
    if failure != "plan-read":
        assert errors and errors[0][0].rel_path == NEW


@pytest.mark.parametrize("change", ["markdown", "embedding"])
def test_move_rechecks_embedding_reuse_at_execution(indexer: Indexer, collection, monkeypatch, change):
    indexer.sync()
    move_entry(indexer.md_dir, OLD, NEW)
    indexer.reload()
    plan = indexer.plan()
    assert plan.to_index[0].metadata_only
    if change == "markdown":
        (indexer.md_dir / "archives/renamed.md").write_text("Changed after planning")
    else:
        collection.update(ids=[OLD], metadatas=[{"embedding": "other-model"}])
    embedded = []
    original = collection._embed
    monkeypatch.setattr(collection, "_embed", lambda **kw: (embedded.append(kw), original(**kw))[1])
    assert indexer.execute(plan).moved == [NEW]
    assert embedded
    assert not indexer.plan().to_index


@pytest.mark.parametrize("change", ["delete", "fingerprint", "engine"])
def test_stale_move_source_fails_safely(indexer: Indexer, collection, change):
    indexer.sync()
    move_entry(indexer.md_dir, OLD, NEW)
    indexer.reload()
    plan = indexer.plan()
    if change == "delete":
        collection.delete(ids=[OLD])
    else:
        collection.update(ids=[OLD], metadatas=[{change: "changed-after-planning"}])
    result = indexer.execute(plan, prune=True)
    assert "move source" in result.failed[NEW]
    assert not result.moved and not result.pruned
    assert NEW not in collection.get()["ids"]
    assert (OLD in collection.get()["ids"]) == (change != "delete")


def test_retag_handles_moves(indexer: Indexer, collection, monkeypatch):
    indexer.sync()
    move_entry(indexer.md_dir, OLD, NEW)
    indexer.reload()
    monkeypatch.setattr(collection, "_embed", lambda **kw: pytest.fail("retag re-embedded"))
    result = indexer.sync(retag=True)
    assert result.moved == [NEW] and result.indexed == ["docs/notes.txt"]
    assert not result.failed


def test_move_and_prune_unrelated_orphan(indexer: Indexer, collection):
    indexer.sync()
    move_entry(indexer.md_dir, OLD, NEW)
    indexer.reload()
    del indexer.manifest.entries["docs/notes.txt"]
    plan = indexer.plan()
    assert plan.orphans == ["docs/notes.txt"]
    result = indexer.execute(plan, prune=True)
    assert result.moved == [NEW] and result.pruned == ["docs/notes.txt"]
    assert collection.get()["ids"] == [NEW]
