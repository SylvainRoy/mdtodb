from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
import pytest
from chromadb.api.types import EmbeddingFunction

from mdtodb import MANIFEST_NAME, CorpusReader, Indexer, Manifest, ReaderError
from mdtodb.sync import md_sha256

def make_tree(tmp_path: Path, files: dict[str, str], manifest_files=None, version=2) -> Path:
    md_dir = tmp_path / "md"
    md_dir.mkdir(parents=True)
    if manifest_files is None:
        manifest_files = {}
        for rel in files:
            key = "fingerprint" if version == 2 else "sha256"
            manifest_files[rel] = {key: f"fp:{rel}", "engine": "fake", "output": rel.rsplit(".", 1)[0] + ".md"}
    for rel, text in files.items():
        out = manifest_files.get(rel, {}).get("output") or rel.rsplit(".", 1)[0] + ".md"
        target = md_dir / out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, "utf-8")
    (md_dir / MANIFEST_NAME).write_text(json.dumps({"version": version, "files": manifest_files}), "utf-8")
    return md_dir

FILES = {
    "personnes/Estelle/assurance/contrat-habitation.pdf": "# Assurance habitation\n\nFranchise élevée. Résiliation possible.\n",
    "personnes/Marc/prets/pret-immobilier.pdf": "# Prêt immobilier\n\nÉchéancier: 1200 par mois.\n",
    "banque/comptes/contrat-compte.pdf": "# Contrat de compte\n\nFrais de tenue de compte.\n",
}

@pytest.fixture
def md_dir(tmp_path: Path) -> Path:
    return make_tree(tmp_path, FILES)

@pytest.fixture
def collection(fake_embedding):
    import uuid

    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        f"reader_{uuid.uuid4().hex[:12]}", embedding_function=fake_embedding
    )

def index(collection, md_dir: Path, ids=None, embedding_name="fake"):
    indexer = Indexer(md_dir, collection, embedding_name=embedding_name)
    indexer.execute(indexer.plan(select=ids))

def reader_for(md_dir, collection=None):
    if collection is None:
        return CorpusReader(md_dir)
    return CorpusReader(md_dir, collection_factory=lambda: collection)

def test_status_not_configured(md_dir: Path):
    status = CorpusReader(md_dir).corpus_status()
    assert status["manifest_documents"] == 3
    assert status["index_status"] == "not_configured"
    assert status["index_documents"] is None
    assert status["indexed_manifest_documents"] is None
    assert status["orphaned_index_documents"] is None
    assert status["embedding_identifiers"] == []
    assert status["people"] == [
        {"value": "Estelle", "count": 1},
        {"value": "Marc", "count": 1},
    ]
    assert status["filetypes"] == [{"value": "pdf", "count": 3}]

def test_status_counts_with_orphan(md_dir: Path, collection):
    index(collection, md_dir, ids=["personnes/Estelle/assurance/contrat-habitation.pdf"])
    collection.upsert(
        ids=["ghost/orphan.pdf"],
        documents=["orphan"],
        metadatas=[{"file": "ghost/orphan.pdf", "fingerprint": "x", "embedding": "fake"}],
    )
    status = reader_for(md_dir, collection).corpus_status()
    assert status["index_status"] == "available"
    assert status["index_documents"] == 2
    assert status["indexed_manifest_documents"] == 1
    assert status["unindexed_manifest_documents"] == 2
    assert status["orphaned_index_documents"] == 1
    assert status["embedding_identifiers"] == ["fake"]
    blob = json.dumps(status)
    assert "Franchise" not in blob and "Échéancier" not in blob

def test_status_unavailable_index(md_dir: Path):
    calls = []

    def factory():
        calls.append(1)
        raise RuntimeError("secret-boom")

    reader = CorpusReader(md_dir, collection_factory=factory)
    status = reader.corpus_status()
    assert status["index_status"] == "unavailable"
    assert status["index_documents"] is None
    assert "secret-boom" not in json.dumps(status)
    reader2 = CorpusReader(md_dir, collection_factory=lambda: None)
    assert reader2.corpus_status()["index_status"] == "unavailable"

def test_list_pagination_and_filters(md_dir: Path, collection):
    index(collection, md_dir)
    reader = reader_for(md_dir, collection)
    page = reader.list_documents(limit=2)
    assert page["total"] == 3
    assert page["offset"] == 0
    assert page["next_offset"] == 2
    ids = [d["document_id"] for d in page["documents"]]
    assert ids == sorted(ids) == sorted(FILES)[:2]
    page2 = reader.list_documents(limit=2, offset=page["next_offset"])
    assert page2["next_offset"] is None
    assert [d["document_id"] for d in page2["documents"]] == sorted(FILES)[2:]

    estelle = reader.list_documents(person="estelle")
    assert [d["document_id"] for d in estelle["documents"]] == [
        "personnes/Estelle/assurance/contrat-habitation.pdf"
    ]
    assert reader.list_documents(person="nobody")["total"] == 0
    assert reader.list_documents(filetype=".PDF")["total"] == 3
    assert reader.list_documents(filetype="txt")["total"] == 0
    all_kw = reader.list_documents(keywords=["prêt", "immobilier"])
    assert [d["document_id"] for d in all_kw["documents"]] == [
        "personnes/Marc/prets/pret-immobilier.pdf"
    ]
    assert reader.list_documents(keywords=["pret", "estelle"])["total"] == 0
    doc = page["documents"][0]
    assert doc["metadata_is_inferred"] is True
    assert doc["index_state"] == "unchecked"
    assert doc["markdown_sha256"] is None
    assert "Franchise" not in json.dumps(page)

def test_custom_output_and_v1_manifest(tmp_path: Path):
    md_dir = make_tree(
        tmp_path,
        {"a/report.pdf": "# custom\n"},
        manifest_files={
            "a/report.pdf": {"sha256": "fp1", "engine": "fake", "output": "renamed/out.md"}
        },
        version=1,
    )
    reader = CorpusReader(md_dir)
    listed = reader.list_documents()
    assert listed["documents"][0]["markdown_path"] == "renamed/out.md"
    assert reader.read_document("a/report.pdf")["excerpt"]["text"] == "# custom\n"

def test_read_document_pagination_and_hash(md_dir: Path, collection):
    index(collection, md_dir)
    reader = reader_for(md_dir, collection)
    doc_id = "personnes/Estelle/assurance/contrat-habitation.pdf"
    res = reader.read_document(doc_id, max_chars=10)
    expected = FILES[doc_id]
    assert res["document"]["markdown_sha256"] == md_sha256(expected)
    assert res["excerpt"]["markdown_sha256"] == md_sha256(expected)
    assert res["excerpt"]["text"] == expected[:10]
    assert res["excerpt"]["start_offset"] == 0
    assert res["excerpt"]["start_line"] == 1 and res["excerpt"]["end_line"] == 1
    assert res["total_chars"] == len(expected)
    assert res["total_lines"] == len(expected.splitlines())
    parts, offset = [], 0
    while True:
        r = reader.read_document(doc_id, offset=offset, max_chars=10)
        parts.append(r["excerpt"]["text"])
        if r["next_offset"] is None:
            break
        offset = r["next_offset"]
    assert "".join(parts) == expected
    eof = reader.read_document(doc_id, offset=len(expected))
    assert eof["excerpt"]["text"] == "" and eof["next_offset"] is None
    with pytest.raises(ReaderError):
        reader.read_document(doc_id, offset=len(expected) + 1)
    with pytest.raises(ReaderError):
        reader.read_document(doc_id, offset=-1)
    with pytest.raises(ReaderError):
        reader.read_document(doc_id, max_chars=0)
    with pytest.raises(ReaderError):
        reader.read_document(doc_id, max_chars=20001)

def test_read_long_single_line(tmp_path: Path):
    long_line = "x" * 50000
    md_dir = make_tree(tmp_path, {"a/big.pdf": long_line})
    reader = CorpusReader(md_dir)
    parts, offset = [], 0
    while True:
        r = reader.read_document("a/big.pdf", offset=offset, max_chars=20000)
        parts.append(r["excerpt"]["text"])
        assert r["excerpt"]["start_line"] == r["excerpt"]["end_line"] == 1
        if r["next_offset"] is None:
            break
        offset = r["next_offset"]
    assert "".join(parts) == long_line
    assert r["total_lines"] == 1

def test_read_crlf_and_empty(tmp_path: Path):
    md_dir = make_tree(tmp_path, {"a/crlf.pdf": "", "a/empty.pdf": ""})
    (md_dir / "a/crlf.md").write_bytes(b"line1\r\nline2\r\n")
    reader = CorpusReader(md_dir)
    res = reader.read_document("a/crlf.pdf")
    assert res["excerpt"]["text"] == "line1\nline2\n"
    assert res["excerpt"]["markdown_sha256"] == md_sha256("line1\nline2\n")
    assert res["total_lines"] == 2
    empty = reader.read_document("a/empty.pdf")
    assert empty["total_chars"] == 0 and empty["total_lines"] == 0
    assert empty["excerpt"]["end_line"] == 1

def test_freshness_states(md_dir: Path, collection):
    index(collection, md_dir)
    reader = reader_for(md_dir, collection)
    doc_id = "banque/comptes/contrat-compte.pdf"
    assert reader.read_document(doc_id)["document"]["index_state"] == "current"

    data = json.loads((md_dir / MANIFEST_NAME).read_text())
    data["files"][doc_id]["fingerprint"] = "changed"
    (md_dir / MANIFEST_NAME).write_text(json.dumps(data))
    assert reader.read_document(doc_id)["document"]["index_state"] == "source_changed"

    data["files"][doc_id]["fingerprint"] = f"fp:{doc_id}"
    (md_dir / MANIFEST_NAME).write_text(json.dumps(data))
    (md_dir / "banque/comptes/contrat-compte.md").write_text("edited\n", "utf-8")
    assert reader.read_document(doc_id)["document"]["index_state"] == "markdown_changed"
    info = reader.list_documents(keywords=["contrat", "compte"])["documents"][0]
    assert info["index_state"] == "unchecked"

def test_unchecked_when_stored_hash_missing(md_dir: Path, collection):
    index(collection, md_dir)
    doc_id = "banque/comptes/contrat-compte.pdf"
    meta = collection.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
    meta["md_sha256"] = None
    collection.update(ids=[doc_id], metadatas=[meta])
    reader = reader_for(md_dir, collection)
    res = reader.read_document(doc_id)
    assert res["document"]["index_state"] == "unchecked"

def test_unindexed_document_lexically_available(md_dir: Path, collection):
    index(collection, md_dir, ids=["banque/comptes/contrat-compte.pdf"])
    reader = reader_for(md_dir, collection)
    res = reader.read_document("personnes/Marc/prets/pret-immobilier.pdf")
    assert res["document"]["index_state"] == "unindexed"
    hits = reader.search_documents("Échéancier", mode="lexical")
    assert [r["document"]["document_id"] for r in hits["results"]] == [
        "personnes/Marc/prets/pret-immobilier.pdf"
    ]

def test_find_literal_not_regex(tmp_path: Path):
    md_dir = make_tree(tmp_path, {"a/doc.pdf": "value a.b and [abc] here.\nsecond line.\n"})
    reader = CorpusReader(md_dir)
    res = reader.find_in_document("a/doc.pdf", "[abc]")
    assert len(res["matches"]) == 1
    m = res["matches"][0]
    text = "value a.b and [abc] here.\nsecond line.\n"
    assert text[m["match_start"] : m["match_end"]] == "[abc]"
    res = reader.find_in_document("a/doc.pdf", ".")
    assert len(res["matches"]) == 3

def test_find_character_offset_pagination(tmp_path: Path):
    text = "prefix clause; more words; clause ends"
    md_dir = make_tree(tmp_path, {"a/doc.pdf": text})
    reader = CorpusReader(md_dir)
    res = reader.find_in_document("a/doc.pdf", "clause", limit=1)
    assert len(res["matches"]) == 1
    assert res["matches"][0]["match_start"] == 7
    assert res["next_offset"] == 13
    res2 = reader.find_in_document("a/doc.pdf", "clause", offset=13, limit=1)
    assert len(res2["matches"]) == 1
    assert res2["matches"][0]["match_start"] == 27
    assert res2["next_offset"] is None

def test_find_iterates_to_eof(tmp_path: Path):
    text = "Été été ETE.\n" * 30
    md_dir = make_tree(tmp_path, {"a/doc.pdf": text})
    reader = CorpusReader(md_dir)
    positions = []
    offset = 0
    while True:
        res = reader.find_in_document("a/doc.pdf", "été", offset=offset, limit=10)
        positions.extend(m["match_start"] for m in res["matches"])
        if res["next_offset"] is None:
            break
        offset = res["next_offset"]
    expected = [i for i in range(len(text)) if text[i : i + 3].lower() == "été"]
    assert positions == expected
    eof = reader.find_in_document("a/doc.pdf", "été", offset=len(text))
    assert eof["matches"] == [] and eof["next_offset"] is None
    with pytest.raises(ReaderError):
        reader.find_in_document("a/doc.pdf", "été", offset=len(text) + 1)

def test_find_bounded_iteration_many_matches(tmp_path: Path):
    text = "a" * 50000
    md_dir = make_tree(tmp_path, {"a/doc.pdf": text})
    reader = CorpusReader(md_dir)
    res = reader.find_in_document("a/doc.pdf", "A", limit=5)
    assert len(res["matches"]) == 5
    assert res["next_offset"] == res["matches"][-1]["match_end"] == 5

def test_find_trailing_newline_offsets(tmp_path: Path):
    text = "abc\n"
    md_dir = make_tree(tmp_path, {"a/doc.pdf": text})
    reader = CorpusReader(md_dir)
    res = reader.find_in_document("a/doc.pdf", "abc")
    m = res["matches"][0]
    assert (m["match_start"], m["match_end"]) == (0, 3)
    assert m["excerpt"]["text"] == text
    assert m["excerpt"]["start_line"] == 1 and m["excerpt"]["end_line"] == 1

@pytest.mark.parametrize(
    "bad_id",
    [
        "../escape.pdf",
        "/abs/path.pdf",
        "a\\b.pdf",
        "a/./b.pdf",
        ".hidden/doc.pdf",
        "a/.hidden.pdf",
        "a//b.pdf",
        "x\x00.pdf",
        "ghost/missing.pdf",
    ],
)
def test_invalid_or_unknown_ids(md_dir: Path, bad_id):
    reader = CorpusReader(md_dir)
    for call in (
        lambda: reader.read_document(bad_id),
        lambda: reader.find_in_document(bad_id, "x"),
    ):
        with pytest.raises(ReaderError):
            call()

def test_unknown_id_existing_file_not_read(md_dir: Path, tmp_path: Path):
    marker = md_dir / "a/secret.md"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("secret", "utf-8")
    reader = CorpusReader(md_dir)
    with pytest.raises(ReaderError):
        reader.read_document("a/secret.pdf")

def test_manifest_entries_validated(tmp_path: Path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret outside", "utf-8")
    cases = [
        {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "../outside.md"}},
        {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": ".hidden/x.md"}},
        {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.txt"}},
        {"../x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.md"}},
    ]
    for files in cases:
        md_dir = make_tree(tmp_path / f"c{len(list(tmp_path.iterdir()))}", {}, manifest_files=files)
        reader = CorpusReader(md_dir)
        with pytest.raises(ReaderError):
            reader.list_documents()
        assert outside.read_text() == "secret outside"

def test_symlink_escape_rejected(tmp_path: Path):
    md_dir = make_tree(
        tmp_path,
        {},
        manifest_files={"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "link/x.md"}},
    )
    secret = tmp_path / "secret.md"
    secret.write_text("top secret", "utf-8")
    os.symlink(tmp_path, md_dir / "link")
    reader = CorpusReader(md_dir)
    with pytest.raises(ReaderError):
        reader.read_document("a/x.pdf")
    real_manifest = (md_dir / MANIFEST_NAME).read_text()
    (md_dir / MANIFEST_NAME).unlink()
    external = tmp_path / "ext-manifest.json"
    external.write_text(real_manifest, "utf-8")
    os.symlink(external, md_dir / MANIFEST_NAME)
    with pytest.raises(ReaderError):
        CorpusReader(md_dir).list_documents()

def test_missing_and_invalid_markdown(md_dir: Path):
    (md_dir / "banque/comptes/contrat-compte.md").unlink()
    (md_dir / "personnes/Marc/prets/pret-immobilier.md").write_bytes(b"\xff\xfe invalid")
    reader = CorpusReader(md_dir)
    for doc_id in (
        "banque/comptes/contrat-compte.pdf",
        "personnes/Marc/prets/pret-immobilier.pdf",
    ):
        with pytest.raises(ReaderError):
            reader.read_document(doc_id)
    hits = reader.search_documents("compte", mode="lexical")
    assert hits["coverage"]["skipped_documents"] == 2
    assert any("incomplete" in w for w in hits["warnings"])
    assert hits["coverage"]["lexically_scanned_documents"] == 1

def test_oversized_document(tmp_path: Path):
    md_dir = make_tree(tmp_path, {"a/big.pdf": "x"})
    big = md_dir / "a/big.md"
    big.write_bytes(b"y" * (16 * 1024 * 1024 + 1))
    reader = CorpusReader(md_dir)
    with pytest.raises(ReaderError):
        reader.read_document("a/big.pdf")
    res = reader.search_documents("y", mode="lexical")
    assert res["coverage"]["skipped_documents"] == 1

def test_directory_output_rejected(tmp_path: Path):
    md_dir = make_tree(
        tmp_path, {}, manifest_files={"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.md"}}
    )
    (md_dir / "a/x.md").mkdir(parents=True)
    reader = CorpusReader(md_dir)
    with pytest.raises(ReaderError):
        reader.read_document("a/x.pdf")

def test_lexical_and_semantics_and_ranking(tmp_path: Path):
    files = {
        "banque/document.pdf": "Le document.\n" + ("remplissage\n" * 300) + "frais et compte loin.\n",
        "compte/autre.pdf": "frais mentions but compte also here\n",
        "random/frais-compte.pdf": "no body tokens at all\n",
    }
    md_dir = make_tree(tmp_path, files)
    reader = CorpusReader(md_dir)
    res = reader.search_documents("frais compte", mode="lexical")
    assert res["coverage"]["eligible_documents"] == 3
    assert res["coverage"]["lexically_scanned_documents"] == 3
    assert res["results"][0]["document"]["document_id"] == "random/frais-compte.pdf"
    assert res["results"][0]["matched_by"] == ["lexical"]
    hit = {r["document"]["document_id"]: r for r in res["results"]}["banque/document.pdf"]
    text = files["banque/document.pdf"]
    expected_start = text.index("frais et compte")
    assert hit["excerpt"]["start_offset"] == expected_start
    assert hit["excerpt"]["text"].startswith("frais et compte")
    assert res["results"][0]["distance"] is None

def test_lexical_accent_and_token_boundaries(md_dir: Path):
    reader = CorpusReader(md_dir)
    res = reader.search_documents("pret", mode="lexical")
    assert [r["document"]["document_id"] for r in res["results"]] == [
        "personnes/Marc/prets/pret-immobilier.pdf"
    ]
    assert reader.search_documents("pre immobilier", mode="lexical")["results"] == []
    assert reader.search_documents("téléphone", mode="lexical")["results"] == []
    with pytest.raises(ReaderError):
        reader.search_documents("!!!", mode="lexical")
    with pytest.raises(ReaderError):
        reader.search_documents("x", mode="bogus")
    with pytest.raises(ReaderError):
        reader.search_documents("x" * 1001, mode="lexical")
    with pytest.raises(ReaderError):
        reader.search_documents("ok", limit=31)

class SpyEmbedding(EmbeddingFunction):
    def __init__(self, name="fake", config=None, fail=None):
        self._name = name
        self._config = config or {}
        self._fail = fail
        self.embed_calls = []

    def __call__(self, input):
        return [[0.1] * 8 for _ in input]

    def embed_query(self, input):
        self.embed_calls.append(list(input))
        if self._fail:
            raise RuntimeError(self._fail)
        return [[0.1] * 8 for _ in input]

    @staticmethod
    def build_from_config(config):
        return SpyEmbedding()

    def name(self):
        return self._name

    def get_config(self):
        return dict(self._config)

def semantic_reader(md_dir, collection, ef=None):
    ef = ef or SpyEmbedding()
    queries = []

    def factory():
        return collection

    def ef_factory():
        return ef, "fake"

    reader = CorpusReader(md_dir, collection_factory=factory, embedding_factory=ef_factory)
    return reader, ef

def test_embedder_not_used_for_non_semantic(md_dir: Path, collection):
    index(collection, md_dir)
    ef = SpyEmbedding()
    called = []
    reader = CorpusReader(
        md_dir,
        collection_factory=lambda: collection,
        embedding_factory=lambda: (called.append(1), (ef, "fake"))[1],
    )
    reader.corpus_status()
    reader.list_documents()
    reader.search_documents("frais", mode="lexical")
    reader.read_document("banque/comptes/contrat-compte.pdf")
    reader.find_in_document("banque/comptes/contrat-compte.pdf", "frais")
    assert called == [] and ef.embed_calls == []

def test_semantic_no_candidates_no_provider_call(md_dir: Path, collection):
    ef = SpyEmbedding()
    reader, _ = semantic_reader(md_dir, collection, ef)
    res = reader.search_documents("anything", mode="semantic")
    assert res["results"] == []
    assert ef.embed_calls == []

def test_semantic_happy_path(md_dir: Path, collection):
    index(collection, md_dir)
    ef = SpyEmbedding()
    reader, ef = semantic_reader(md_dir, collection, ef)
    captured = {}
    orig_query = collection.query

    def spy(**kwargs):
        captured.update(kwargs)
        return orig_query(**kwargs)

    collection.query = spy
    res = reader.search_documents("assurance", mode="semantic", limit=2)
    assert ef.embed_calls == [["assurance"]]
    assert set(captured["ids"]) <= set(FILES)
    assert captured["n_results"] <= min(60, len(FILES))
    assert "documents" not in captured["include"]
    assert res["methods_used"] == ["semantic"]
    for r in res["results"]:
        assert r["matched_by"] == ["semantic"]
        assert isinstance(r["distance"], float)
        assert r["excerpt"]["text"]
    dists = [r["distance"] for r in res["results"]]
    assert dists == sorted(dists)

def test_orphan_never_returned(md_dir: Path, collection):
    index(collection, md_dir, ids=["banque/comptes/contrat-compte.pdf"])
    collection.upsert(
        ids=["ghost/orphan.pdf"],
        documents=["orphan body"],
        metadatas=[{"file": "ghost/orphan.pdf", "fingerprint": "x", "embedding": "fake"}],
    )
    reader, ef = semantic_reader(md_dir, collection)
    res = reader.search_documents("orphan", mode="semantic", limit=30)
    assert all(r["document"]["document_id"] != "ghost/orphan.pdf" for r in res["results"])

def test_semantic_mismatched_embedding_fails_before_query(md_dir: Path, collection):
    index(collection, md_dir, embedding_name="other-embedding")
    ef = SpyEmbedding()
    reader, ef = semantic_reader(md_dir, collection, ef)
    with pytest.raises(ReaderError):
        reader.search_documents("x", mode="semantic")
    assert ef.embed_calls == []

def test_semantic_mismatched_collection_config(md_dir: Path, collection):
    index(collection, md_dir)
    ef = SpyEmbedding(name="different")
    reader = CorpusReader(
        md_dir,
        collection_factory=lambda: collection,
        embedding_factory=lambda: (ef, "fake"),
    )
    with pytest.raises(ReaderError):
        reader.search_documents("x", mode="semantic")
    assert ef.embed_calls == []

def test_semantic_provider_failure_sanitized(md_dir: Path, collection):
    index(collection, md_dir)
    ef = SpyEmbedding(fail="sk-SENTINEL-SECRET")
    reader, ef = semantic_reader(md_dir, collection, ef)
    with pytest.raises(ReaderError) as excinfo:
        reader.search_documents("x", mode="semantic")
    assert "SENTINEL" not in str(excinfo.value)

def test_semantic_index_unavailable(md_dir: Path):
    reader = CorpusReader(
        md_dir,
        collection_factory=lambda: (_ for _ in ()).throw(RuntimeError("no")),
        embedding_factory=lambda: (SpyEmbedding(), "fake"),
    )
    with pytest.raises(ReaderError):
        reader.search_documents("x", mode="semantic")

def test_hybrid_falls_back_to_lexical(md_dir: Path, collection):
    index(collection, md_dir)
    ef = SpyEmbedding(fail="boom")
    reader, _ = semantic_reader(md_dir, collection, ef)
    res = reader.search_documents("frais", mode="hybrid")
    assert res["methods_used"] == ["lexical"]
    assert any("fell back" in w for w in res["warnings"])
    assert res["results"]

def test_rrf_merge_exact():
    reader = CorpusReader("/nonexistent")

    def hit(doc_id, method):
        return {
            "document_id": doc_id,
            "document": {"document_id": doc_id},
            "excerpt": None,
            "matched_by": [method],
            "distance": None,
        }

    lex = [hit("a", "lexical"), hit("b", "lexical")]
    sem = [hit("b", "semantic"), hit("c", "semantic")]
    merged = reader._hybrid_merge(lex, sem, 10)
    assert [h["document_id"] for h in merged] == ["b", "a", "c"]
    assert merged[0]["matched_by"] == ["lexical", "semantic"]

def test_no_writes_to_collection_or_files(md_dir: Path, collection, monkeypatch):
    index(collection, md_dir)
    manifest_bytes = (md_dir / MANIFEST_NAME).read_bytes()
    md_bytes = {p: p.read_bytes() for p in md_dir.rglob("*.md")}
    before = collection.get(include=["metadatas", "documents"])
    for method in ("add", "upsert", "update", "delete", "modify"):
        monkeypatch.setattr(
            collection, method, lambda *a, **k: pytest.fail(f"collection.{method} called")
        )
    monkeypatch.setattr(Manifest, "save", lambda *a, **k: pytest.fail("save called"), raising=False)
    reader = reader_for(md_dir, collection)
    reader.corpus_status()
    reader.list_documents()
    reader.search_documents("frais", mode="lexical")
    reader.read_document("banque/comptes/contrat-compte.pdf")
    reader.find_in_document("banque/comptes/contrat-compte.pdf", "frais")
    assert (md_dir / MANIFEST_NAME).read_bytes() == manifest_bytes
    assert {p: p.read_bytes() for p in md_dir.rglob("*.md")} == md_bytes
    assert collection.get(include=["metadatas", "documents"]) == before

def test_updates_visible_next_call(md_dir: Path, collection):
    index(collection, md_dir)
    reader = reader_for(md_dir, collection)
    assert reader.corpus_status()["manifest_documents"] == 3
    data = json.loads((md_dir / MANIFEST_NAME).read_text())
    data["files"]["new/doc.pdf"] = {"fingerprint": "f", "engine": "e", "output": "new/doc.md"}
    (md_dir / MANIFEST_NAME).write_text(json.dumps(data))
    (md_dir / "new").mkdir()
    (md_dir / "new/doc.md").write_text("new text\n", "utf-8")
    assert reader.corpus_status()["manifest_documents"] == 4
    assert reader.corpus_status()["unindexed_manifest_documents"] == 1

def test_manifest_malformed_structures(tmp_path: Path):
    base = {"version": 2, "files": {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.md"}}}
    cases = [
        {},
        {"files": []},
        {"version": 2},
        {"version": 3, "files": {}},
        {"version": "2", "files": {}},
        {"version": True, "files": {}},
        {"version": 2, "files": {"a/x.pdf": "not-a-dict"}},
        {"version": 2, "files": {"a/x.pdf": {"engine": "e", "output": "a/x.md"}}},
        {"version": 2, "files": {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": 5}}},
        {"version": 2, "files": {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.md", "size": -1}}},
        {"version": 2, "files": {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.md", "size": True}}},
        {"version": 2, "files": {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.md", "mtime": float("nan")}}},
        {"version": 2, "files": {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.md", "generated_at": "now"}}},
        {"version": 2, "files": {"a/./x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.md"}}},
        {"version": 2, "files": {"a//x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/x.md"}}},
        {"version": 2, "files": {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "a/./x.md"}}},
        {"version": 2, "files": {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "a//x.md"}}},
        {"version": 2, "files": {"a/x.pdf": {"fingerprint": "f", "engine": "e", "output": "/abs/x.md"}}},
    ]
    for i, manifest in enumerate(cases):
        md_dir = tmp_path / f"case{i}" / "md"
        md_dir.mkdir(parents=True)
        (md_dir / MANIFEST_NAME).write_text(json.dumps(manifest), "utf-8")
        reader = CorpusReader(md_dir)
        with pytest.raises(ReaderError):
            reader.corpus_status()


def test_manifest_empty_is_valid(tmp_path: Path):
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / MANIFEST_NAME).write_text(json.dumps({"version": 2, "files": {}}), "utf-8")
    status = CorpusReader(md_dir).corpus_status()
    assert status["manifest_documents"] == 0


def test_manifest_entry_extra_fields_ok(tmp_path: Path):
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "x.md").write_text("hi\n", "utf-8")
    entry = {"fingerprint": "f", "engine": "e", "output": "x.md", "unknown_field": {"nested": 1}}
    (md_dir / MANIFEST_NAME).write_text(json.dumps({"version": 2, "files": {"x.pdf": entry}}), "utf-8")
    assert CorpusReader(md_dir).corpus_status()["manifest_documents"] == 1


class MisalignedCollection:
    def __init__(self, page):
        self._page = page

    def get(self, **kwargs):
        return self._page


def test_index_misaligned_page_fails_closed(md_dir: Path):
    bad_pages = [
        {"ids": ["a"], "metadatas": [{}, {}]},
        {"ids": "a/x.pdf", "metadatas": [{}]},
        "not-a-dict",
    ]
    for page in bad_pages:
        reader = CorpusReader(md_dir, collection_factory=lambda: MisalignedCollection(page))
        status = reader.corpus_status()
        assert status["index_status"] == "unavailable"
        assert status["index_documents"] is None
    ok = CorpusReader(
        md_dir, collection_factory=lambda: MisalignedCollection({"ids": ["a"], "metadatas": None})
    )
    status = ok.corpus_status()
    assert status["index_status"] == "available"
    assert status["index_documents"] == 1


def test_semantic_unreadable_markdown_warns(md_dir: Path, collection):
    index(collection, md_dir)
    (md_dir / "banque/comptes/contrat-compte.md").unlink()
    (md_dir / "personnes/Marc/prets/pret-immobilier.md").write_bytes(b"\xff invalid")
    reader, ef = semantic_reader(md_dir, collection)
    res = reader.search_documents("assurance", mode="semantic", limit=10)
    assert res["coverage"]["skipped_documents"] == 2
    assert res["coverage"]["semantic_candidate_documents"] == 3
    assert any("incomplete" in w for w in res["warnings"])
    by_id = {r["document"]["document_id"]: r for r in res["results"]}
    assert by_id["banque/comptes/contrat-compte.pdf"]["excerpt"] is None
    assert by_id["banque/comptes/contrat-compte.pdf"]["document"]["markdown_sha256"] is None
    assert by_id["personnes/Estelle/assurance/contrat-habitation.pdf"]["excerpt"] is not None


def test_semantic_candidate_coverage_warning(md_dir: Path, collection):
    index(collection, md_dir, ids=["banque/comptes/contrat-compte.pdf"])
    reader, ef = semantic_reader(md_dir, collection)
    res = reader.search_documents("compte", mode="semantic", limit=10)
    assert res["coverage"]["semantic_candidate_documents"] == 1
    assert res["coverage"]["eligible_documents"] == 3
    assert any("not indexed" in w for w in res["warnings"])
    assert all(
        r["document"]["document_id"] == "banque/comptes/contrat-compte.pdf"
        for r in res["results"]
    )


def test_hybrid_skipped_counted_once(md_dir: Path, collection):
    index(collection, md_dir)
    (md_dir / "banque/comptes/contrat-compte.md").unlink()
    reader, ef = semantic_reader(md_dir, collection)
    res = reader.search_documents("compte", mode="hybrid", limit=10)
    assert res["coverage"]["skipped_documents"] == 1


class BrokenConfigEmbedding(SpyEmbedding):
    def get_config(self):
        raise RuntimeError("cannot serialize")


def test_embedding_config_unverifiable_fails(md_dir: Path, collection):
    index(collection, md_dir)
    ef = BrokenConfigEmbedding()
    reader = CorpusReader(
        md_dir,
        collection_factory=lambda: collection,
        embedding_factory=lambda: (ef, "fake"),
    )
    with pytest.raises(ReaderError):
        reader.search_documents("x", mode="semantic")
    assert ef.embed_calls == []
    res = reader.search_documents("frais", mode="hybrid")
    assert res["methods_used"] == ["lexical"]
    assert any("fell back" in w for w in res["warnings"])
    assert ef.embed_calls == []


def test_rrf_merge_does_not_mutate_inputs():
    reader = CorpusReader("/nonexistent")

    def hit(doc_id, method):
        return {
            "document_id": doc_id,
            "document": {"document_id": doc_id},
            "excerpt": None,
            "matched_by": [method],
            "distance": None,
        }

    lex = [hit("a", "lexical"), hit("b", "lexical")]
    sem = [hit("b", "semantic"), hit("c", "semantic")]
    lex_before = [dict(h, matched_by=list(h["matched_by"])) for h in lex]
    sem_before = [dict(h, matched_by=list(h["matched_by"])) for h in sem]
    merged = reader._hybrid_merge(lex, sem, 10)
    assert [h["document_id"] for h in merged] == ["b", "a", "c"]
    assert merged[0]["matched_by"] == ["lexical", "semantic"]
    assert lex == lex_before and sem == sem_before


def test_search_result_public_keys(md_dir: Path):
    reader = CorpusReader(md_dir)
    res = reader.search_documents("frais", mode="lexical")
    for r in res["results"]:
        assert set(r) == {"document", "excerpt", "matched_by", "distance"}


def test_total_lines_citation_consistent(tmp_path: Path):
    text = "one\ftwo\u2028three\nfour"
    md_dir = make_tree(tmp_path, {"a/doc.pdf": text})
    reader = CorpusReader(md_dir)
    res = reader.read_document("a/doc.pdf")
    assert res["total_lines"] == 2
    res2 = reader.read_document("a/doc.pdf", offset=text.index("three"))
    assert res2["excerpt"]["start_line"] == 1
    res3 = reader.read_document("a/doc.pdf", offset=text.index("four"))
    assert res3["excerpt"]["start_line"] == 2
