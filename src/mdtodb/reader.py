"""Read-only, per-call-fresh access to a pdftomd/mdtodb corpus.

``CorpusReader`` exposes manifest-listed Markdown documents plus optional
Chroma index metadata. It performs no writes to the manifest, the Markdown
tree, or the collection, and it only touches the embedding provider for
explicit semantic/hybrid searches. Every operation reloads the manifest and
the index metadata so manual ingestion updates are visible immediately.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Callable, Literal

from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction

from .manifest import MANIFEST_NAME, Manifest, ManifestEntry, markdown_path_for
from .metadata import filetype_for, keywords_for, normalize_keyword, person_for
from .sync import md_sha256

MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_QUERY_CHARS = 1000
LIST_LIMIT = (1, 100)
SEARCH_LIMIT = (1, 30)
FIND_LIMIT = (1, 30)
READ_MAX_CHARS = (1, 20000)
INDEX_PAGE = 500
FIND_CONTEXT_BEFORE = 200
EXCERPT_CHARS = 1200
_RRF_K = 60
_TOKEN_RE = re.compile(r"[^\W_]+")
_EMBEDDING_CONFIG_KEYS = ("model_name", "task_type", "dimension", "vertexai", "project", "location")


class ReaderError(Exception):
    """Expected, sanitized failure of a reader operation."""


@dataclass
class DocumentInfo:
    document_id: str
    markdown_path: str
    filetype: str
    person: str | None
    keywords: list[str]
    metadata_is_inferred: bool
    engine: str
    fingerprint: str
    generated_at: float | None
    indexed_at: float | None
    embedding: str | None
    index_state: str
    markdown_sha256: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Excerpt:
    document_id: str
    markdown_sha256: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Snapshot:
    manifest: Manifest
    collection: Collection | None
    index_status: str
    index_metadata: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _check_range(name: str, value: int, lo: int, hi: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        raise ReaderError(f"{name} must be between {lo} and {hi}")


def _check_offset(offset: int) -> None:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ReaderError("offset must be a non-negative integer")


def _check_query(query: str) -> str:
    if not isinstance(query, str):
        raise ReaderError("query must be a string")
    query = query.strip()
    if not 1 <= len(query) <= MAX_QUERY_CHARS:
        raise ReaderError(f"query must be 1..{MAX_QUERY_CHARS} characters")
    return query


def _valid_rel_path(value: str) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    return all(part and not part.startswith(".") for part in value.split("/"))


class CorpusReader:
    """Read-only view over a pdftomd manifest, its Markdown and a Chroma index.

    ``collection_factory`` and ``embedding_factory`` are invoked lazily, per
    operation, so failures are never cached and the embedding provider is
    only resolved for semantic requests.
    """

    def __init__(
        self,
        markdown_root: str | Path,
        *,
        collection_factory: Callable[[], Collection] | None = None,
        embedding_factory: Callable[[], tuple[EmbeddingFunction, str]] | None = None,
    ) -> None:
        self._root = Path(markdown_root).expanduser().resolve()
        self._collection_factory = collection_factory
        self._embedding_factory = embedding_factory


    def _load_manifest(self) -> Manifest:
        manifest_path = self._root / MANIFEST_NAME
        try:
            resolved = manifest_path.resolve()
        except OSError:
            raise ReaderError("Corpus manifest is not readable") from None
        if not resolved.is_relative_to(self._root) or not resolved.is_file():
            raise ReaderError("Corpus manifest not found under the Markdown root")
        try:
            size = resolved.stat().st_size
        except OSError:
            raise ReaderError("Corpus manifest is not readable") from None
        if size > MAX_MANIFEST_BYTES:
            raise ReaderError("Corpus manifest exceeds the maximum size")
        try:
            with open(resolved, "rb") as handle:
                data = handle.read(MAX_MANIFEST_BYTES + 1)
        except OSError:
            raise ReaderError("Corpus manifest is not readable") from None
        if len(data) > MAX_MANIFEST_BYTES:
            raise ReaderError("Corpus manifest exceeds the maximum size")
        try:
            parsed = json.loads(data.decode("utf-8"))
        except Exception:
            raise ReaderError("Corpus manifest is not valid") from None
        self._validate_manifest(parsed)
        try:
            manifest = Manifest.from_dict(parsed)
        except Exception:
            raise ReaderError("Corpus manifest is not valid") from None
        for document_id, entry in manifest.entries.items():
            if not _valid_rel_path(document_id) or not self._valid_output(entry, document_id):
                raise ReaderError("Corpus manifest contains an invalid entry")
        return manifest

    @staticmethod
    def _validate_manifest(parsed) -> None:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("files"), dict):
            raise ReaderError("Corpus manifest is not valid")
        version = parsed.get("version")
        if version is not None and (
            not isinstance(version, int) or isinstance(version, bool) or version not in (1, 2)
        ):
            raise ReaderError("Corpus manifest is not valid")
        for key, value in parsed["files"].items():
            if not isinstance(value, dict):
                raise ReaderError("Corpus manifest is not valid")
            fingerprint = value.get("fingerprint", value.get("sha256"))
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ReaderError("Corpus manifest is not valid")
            if not isinstance(value.get("engine"), str):
                raise ReaderError("Corpus manifest is not valid")
            output = value.get("output")
            if output is not None and not isinstance(output, str):
                raise ReaderError("Corpus manifest is not valid")
            size = value.get("size")
            if size is not None and (
                not isinstance(size, int) or isinstance(size, bool) or size < 0
            ):
                raise ReaderError("Corpus manifest is not valid")
            for field_name in ("mtime", "generated_at"):
                field_value = value.get(field_name)
                if field_value is not None and (
                    not isinstance(field_value, (int, float))
                    or isinstance(field_value, bool)
                    or not math.isfinite(field_value)
                    or field_value < 0
                ):
                    raise ReaderError("Corpus manifest is not valid")

    def _open_index(self) -> tuple[Collection | None, str, str | None]:
        if self._collection_factory is None:
            return None, "not_configured", None
        try:
            collection = self._collection_factory()
        except Exception:
            return None, "unavailable", "Document index is not available"
        if collection is None:
            return None, "unavailable", "Document index is not available"
        return collection, "available", None

    def _index_metadata(self, collection: Collection) -> dict[str, dict]:
        state: dict[str, dict] = {}
        offset = 0
        while True:
            page = collection.get(limit=INDEX_PAGE, offset=offset, include=["metadatas"])
            if not isinstance(page, dict):
                raise ReaderError("Document index metadata could not be read")
            ids = page.get("ids")
            metadatas = page.get("metadatas")
            if not isinstance(ids, list):
                raise ReaderError("Document index metadata could not be read")
            if metadatas is None:
                metadatas = [None] * len(ids)
            if not isinstance(metadatas, list) or len(metadatas) != len(ids):
                raise ReaderError("Document index metadata could not be read")
            for doc_id, meta in zip(ids, metadatas):
                state[doc_id] = meta or {}
            if len(ids) < INDEX_PAGE:
                return state
            offset += INDEX_PAGE

    def _snapshot(self, *, with_index: bool = True) -> _Snapshot:
        manifest = self._load_manifest()
        snap = _Snapshot(manifest=manifest, collection=None, index_status="not_configured")
        if not with_index:
            return snap
        collection, status, warning = self._open_index()
        snap.index_status = status
        if warning:
            snap.warnings.append(warning)
        if collection is None:
            return snap
        try:
            snap.index_metadata = self._index_metadata(collection)
        except Exception:
            snap.index_status = "unavailable"
            snap.warnings.append("Document index metadata could not be read")
            return snap
        snap.collection = collection
        return snap


    def _valid_output(self, entry: ManifestEntry, document_id: str) -> bool:
        output = entry.output or markdown_path_for(document_id)
        return _valid_rel_path(output) and PurePosixPath(output).suffix == ".md"

    def _entry(self, snap: _Snapshot, document_id: str) -> ManifestEntry:
        if not _valid_rel_path(document_id):
            raise ReaderError("Invalid document id")
        entry = snap.manifest.entries.get(document_id)
        if entry is None:
            raise ReaderError("Unknown document id")
        return entry

    def _markdown_file(self, document_id: str, entry: ManifestEntry) -> Path:
        output = entry.output or markdown_path_for(document_id)
        if not _valid_rel_path(output) or PurePosixPath(output).suffix != ".md":
            raise ReaderError("Document has no Markdown output")
        try:
            resolved = (self._root / output).resolve()
            is_file = resolved.is_file()
        except OSError:
            raise ReaderError("Document could not be read") from None
        if not resolved.is_relative_to(self._root):
            raise ReaderError("Document output is outside the Markdown root")
        if not is_file:
            raise ReaderError("Document Markdown is missing")
        return resolved

    def _read_markdown(self, path: Path) -> str:
        try:
            with open(path, "rb") as handle:
                data = handle.read(MAX_DOCUMENT_BYTES + 1)
        except OSError:
            raise ReaderError("Document could not be read") from None
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ReaderError("Document exceeds the maximum size")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ReaderError("Document is not valid UTF-8") from None
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _read_document_text(self, document_id: str, entry: ManifestEntry) -> str:
        return self._read_markdown(self._markdown_file(document_id, entry))


    def _index_state(
        self,
        snap: _Snapshot,
        entry: ManifestEntry,
        meta: dict | None,
        md_hash: str | None,
    ) -> str:
        if snap.index_status != "available":
            return "unavailable"
        if meta is None:
            return "unindexed"
        if meta.get("fingerprint") != entry.fingerprint:
            return "source_changed"
        if not meta.get("md_sha256"):
            return "unchecked"
        if md_hash is None:
            return "unchecked"
        if meta["md_sha256"] != md_hash:
            return "markdown_changed"
        return "current"

    def _document_info(
        self,
        snap: _Snapshot,
        document_id: str,
        md_hash: str | None = None,
    ) -> DocumentInfo:
        entry = snap.manifest.entries[document_id]
        meta = snap.index_metadata.get(document_id)
        keywords = {normalize_keyword(k) for k in keywords_for(document_id)}
        if meta:
            for kw in meta.get("keywords") or []:
                keywords.add(normalize_keyword(str(kw)))
        return DocumentInfo(
            document_id=document_id,
            markdown_path=entry.output or markdown_path_for(document_id),
            filetype=filetype_for(document_id),
            person=person_for(document_id),
            keywords=sorted(keywords),
            metadata_is_inferred=True,
            engine=entry.engine,
            fingerprint=entry.fingerprint,
            generated_at=entry.generated_at or None,
            indexed_at=meta.get("indexed_at") if meta else None,
            embedding=meta.get("embedding") if meta else None,
            index_state=self._index_state(snap, entry, meta, md_hash),
            markdown_sha256=md_hash,
        )

    def _doc_keywords(self, snap: _Snapshot, document_id: str) -> set[str]:
        keywords = {normalize_keyword(k) for k in keywords_for(document_id)}
        meta = snap.index_metadata.get(document_id)
        if meta:
            for kw in meta.get("keywords") or []:
                keywords.add(normalize_keyword(str(kw)))
        return keywords

    def _filtered_ids(
        self,
        snap: _Snapshot,
        person: str | None,
        keywords: list[str] | None,
        filetype: str | None,
    ) -> list[str]:
        person_norm = normalize_keyword(person) if person is not None else None
        wanted_keywords = (
            {normalize_keyword(k) for k in keywords} if keywords is not None else None
        )
        filetype_norm = filetype.lstrip(".").lower() if filetype is not None else None
        out = []
        for document_id in sorted(snap.manifest.entries):
            if person_norm is not None:
                doc_person = person_for(document_id)
                if doc_person is None or normalize_keyword(doc_person) != person_norm:
                    continue
            if filetype_norm is not None and filetype_for(document_id) != filetype_norm:
                continue
            if wanted_keywords is not None and not wanted_keywords <= self._doc_keywords(snap, document_id):
                continue
            out.append(document_id)
        return out


    def _excerpt(self, document_id: str, text: str, md_hash: str, start: int, end: int) -> Excerpt:
        start = max(0, start)
        end = min(len(text), end)
        return Excerpt(
            document_id=document_id,
            markdown_sha256=md_hash,
            start_line=text.count("\n", 0, start) + 1,
            end_line=text.count("\n", 0, max(start, end - 1)) + 1,
            start_offset=start,
            end_offset=end,
            text=text[start:end],
        )


    def corpus_status(self) -> dict:
        snap = self._snapshot()
        warnings = list(snap.warnings)
        manifest_ids = set(snap.manifest.entries)
        index_ids = set(snap.index_metadata)
        index_available = snap.index_status == "available"

        people: dict[str, int] = {}
        filetypes: dict[str, int] = {}
        for document_id in manifest_ids:
            person = person_for(document_id)
            if person:
                people[person] = people.get(person, 0) + 1
            filetype = filetype_for(document_id)
            if filetype:
                filetypes[filetype] = filetypes.get(filetype, 0) + 1

        def facet(counts: dict[str, int]) -> list[dict]:
            return [
                {"value": value, "count": count}
                for value, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ]

        return {
            "manifest_documents": len(manifest_ids),
            "index_status": snap.index_status,
            "index_documents": len(index_ids) if index_available else None,
            "indexed_manifest_documents": len(manifest_ids & index_ids) if index_available else None,
            "unindexed_manifest_documents": len(manifest_ids - index_ids) if index_available else None,
            "orphaned_index_documents": len(index_ids - manifest_ids) if index_available else None,
            "embedding_identifiers": sorted(
                {str(m["embedding"]) for m in snap.index_metadata.values() if m.get("embedding")}
            ),
            "people": facet(people),
            "filetypes": facet(filetypes),
            "warnings": warnings,
        }

    def list_documents(
        self,
        *,
        person: str | None = None,
        keywords: list[str] | None = None,
        filetype: str | None = None,
        offset: int = 0,
        limit: int = 25,
    ) -> dict:
        _check_offset(offset)
        _check_range("limit", limit, *LIST_LIMIT)
        snap = self._snapshot()
        ids = self._filtered_ids(snap, person, keywords, filetype)
        page = ids[offset : offset + limit]
        next_offset = offset + limit if offset + limit < len(ids) else None
        return {
            "documents": [self._document_info(snap, i).to_dict() for i in page],
            "total": len(ids),
            "offset": offset,
            "next_offset": next_offset,
            "warnings": list(snap.warnings),
        }

    def read_document(self, document_id: str, *, offset: int = 0, max_chars: int = 12000) -> dict:
        _check_offset(offset)
        _check_range("max_chars", max_chars, *READ_MAX_CHARS)
        snap = self._snapshot()
        entry = self._entry(snap, document_id)
        text = self._read_document_text(document_id, entry)
        if offset > len(text):
            raise ReaderError("offset is beyond the end of the document")
        end = min(len(text), offset + max_chars)
        md_hash = md_sha256(text)
        return {
            "document": self._document_info(snap, document_id, md_hash).to_dict(),
            "excerpt": self._excerpt(document_id, text, md_hash, offset, end).to_dict(),
            "total_chars": len(text),
            "total_lines": 0 if not text else text.count("\n") + (0 if text.endswith("\n") else 1),
            "next_offset": end if end < len(text) else None,
            "warnings": list(snap.warnings),
        }

    def find_in_document(
        self,
        document_id: str,
        query: str,
        *,
        offset: int = 0,
        limit: int = 10,
    ) -> dict:
        _check_offset(offset)
        _check_range("limit", limit, *FIND_LIMIT)
        query = _check_query(query)
        snap = self._snapshot()
        entry = self._entry(snap, document_id)
        text = self._read_document_text(document_id, entry)
        md_hash = md_sha256(text)
        if offset > len(text):
            raise ReaderError("offset is beyond the end of the document")
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        page = list(islice(pattern.finditer(text, offset), limit + 1))
        more = len(page) > limit
        page = page[:limit]
        out = []
        for match in page:
            start = max(0, match.start() - FIND_CONTEXT_BEFORE)
            end = min(len(text), start + EXCERPT_CHARS)
            out.append(
                {
                    "match_start": match.start(),
                    "match_end": match.end(),
                    "excerpt": self._excerpt(document_id, text, md_hash, start, end).to_dict(),
                }
            )
        return {
            "document": self._document_info(snap, document_id, md_hash).to_dict(),
            "matches": out,
            "next_offset": page[-1].end() if more else None,
            "warnings": list(snap.warnings),
        }


    def search_documents(
        self,
        query: str,
        *,
        mode: Literal["lexical", "semantic", "hybrid"] = "lexical",
        person: str | None = None,
        keywords: list[str] | None = None,
        filetype: str | None = None,
        limit: int = 10,
    ) -> dict:
        if mode not in ("lexical", "semantic", "hybrid"):
            raise ReaderError("mode must be one of 'lexical', 'semantic', 'hybrid'")
        _check_range("limit", limit, *SEARCH_LIMIT)
        query = _check_query(query)
        snap = self._snapshot()
        eligible = self._filtered_ids(snap, person, keywords, filetype)
        warnings = list(snap.warnings)
        query_tokens = set(_TOKEN_RE.findall(normalize_keyword(query)))
        if mode in ("lexical", "hybrid") and not query_tokens:
            raise ReaderError("query has no searchable terms")

        lex_hits: list[dict] = []
        scanned = 0
        unreadable: set[str] = set()
        if mode in ("lexical", "hybrid"):
            lex_hits, scanned, lex_skipped = self._lexical_scan(snap, eligible, query_tokens)
            unreadable |= set(lex_skipped)

        sem_hits: list[dict] = []
        sem_ok = False
        if mode in ("semantic", "hybrid") and snap.index_status == "available":
            candidate_count = len(set(eligible) & set(snap.index_metadata))
            if len(eligible) > candidate_count:
                warnings.append(
                    f"{len(eligible) - candidate_count} eligible document(s) are not indexed and cannot be searched semantically"
                )
        if mode in ("semantic", "hybrid"):
            try:
                sem_hits, sem_unreadable = self._semantic_search(snap, eligible, query, limit)
                sem_ok = True
                unreadable |= set(sem_unreadable)
            except ReaderError as exc:
                if mode == "semantic":
                    raise
                warnings.append(str(exc) + "; fell back to lexical results only")

        if unreadable:
            shown = ", ".join(sorted(unreadable)[:20])
            warnings.append(
                f"Search coverage is incomplete: {len(unreadable)} document(s) could not be read ({shown})"
            )

        if mode == "lexical":
            methods = ["lexical"]
            ranked = lex_hits
        elif mode == "semantic":
            methods = ["semantic"]
            ranked = sem_hits
        else:
            methods = ["lexical"] + (["semantic"] if sem_ok else [])
            ranked = self._hybrid_merge(lex_hits, sem_hits, limit)

        public = [
            {
                "document": hit["document"],
                "excerpt": hit["excerpt"],
                "matched_by": hit["matched_by"],
                "distance": hit["distance"],
            }
            for hit in ranked[:limit]
        ]
        return {
            "mode_requested": mode,
            "methods_used": methods,
            "results": public,
            "coverage": {
                "eligible_documents": len(eligible),
                "lexically_scanned_documents": scanned if mode in ("lexical", "hybrid") else None,
                "skipped_documents": len(unreadable),
                "semantic_candidate_documents": (
                    len(set(eligible) & set(snap.index_metadata))
                    if mode in ("semantic", "hybrid") and snap.index_status == "available"
                    else None
                ),
            },
            "warnings": warnings,
        }

    def _lexical_scan(
        self, snap: _Snapshot, eligible: list[str], query_tokens: set[str]
    ) -> tuple[list[dict], int, list[str]]:
        hits = []
        skipped: list[str] = []
        scanned = 0
        for document_id in eligible:
            entry = snap.manifest.entries[document_id]
            try:
                text = self._read_document_text(document_id, entry)
            except ReaderError:
                skipped.append(document_id)
                continue
            scanned += 1
            path_tokens = set(_TOKEN_RE.findall(normalize_keyword(document_id)))
            body_tokens = set(_TOKEN_RE.findall(normalize_keyword(text)))
            if not query_tokens <= (path_tokens | body_tokens):
                continue
            md_hash = md_sha256(text)
            start = self._first_token_offset(text, query_tokens)
            hits.append(
                {
                    "sort_key": (-len(query_tokens & path_tokens), document_id),
                    "document_id": document_id,
                    "document": self._document_info(snap, document_id, md_hash).to_dict(),
                    "excerpt": self._excerpt(
                        document_id, text, md_hash, start, start + EXCERPT_CHARS
                    ).to_dict(),
                    "matched_by": ["lexical"],
                    "distance": None,
                }
            )
        hits.sort(key=lambda h: h["sort_key"])
        return hits, scanned, skipped

    @staticmethod
    def _first_token_offset(text: str, query_tokens: set[str]) -> int:
        offset = 0
        for line in text.split("\n"):
            if query_tokens & set(_TOKEN_RE.findall(normalize_keyword(line))):
                return offset
            offset += len(line) + 1
        return 0

    def _semantic_search(
        self, snap: _Snapshot, eligible: list[str], query: str, limit: int
    ) -> tuple[list[dict], list[str]]:
        if snap.index_status != "available" or snap.collection is None:
            raise ReaderError("Semantic search requires the document index, which is unavailable")
        candidates = sorted(set(eligible) & set(snap.index_metadata))
        if not candidates:
            return [], []
        candidate_set = set(candidates)
        if self._embedding_factory is None:
            raise ReaderError("Semantic search is not configured")
        try:
            ef, identifier = self._embedding_factory()
        except ReaderError:
            raise
        except Exception:
            raise ReaderError("Semantic search is unavailable") from None
        self._check_embedding_config(snap.collection, ef)
        for doc_id in candidates:
            if snap.index_metadata[doc_id].get("embedding") != identifier:
                raise ReaderError("Indexed embeddings do not match the configured embedding")
        pool_size = min(60, max(10, limit * 3))
        try:
            response = snap.collection.query(
                query_embeddings=ef.embed_query(input=[query]),
                ids=candidates,
                n_results=min(pool_size, len(candidates)),
                include=["metadatas", "distances"],
            )
        except ReaderError:
            raise
        except Exception:
            raise ReaderError("Semantic search failed") from None
        id_rows = response.get("ids") or [[]]
        dist_rows = response.get("distances") or [[]]
        hits = []
        unreadable: list[str] = []
        for doc_id, distance in zip(id_rows[0], dist_rows[0]):
            if doc_id not in candidate_set:
                continue
            entry = snap.manifest.entries[doc_id]
            md_hash = None
            excerpt = None
            try:
                text = self._read_document_text(doc_id, entry)
                md_hash = md_sha256(text)
                excerpt = self._excerpt(doc_id, text, md_hash, 0, EXCERPT_CHARS).to_dict()
            except ReaderError:
                unreadable.append(doc_id)
            hits.append(
                {
                    "document_id": doc_id,
                    "document": self._document_info(snap, doc_id, md_hash).to_dict(),
                    "excerpt": excerpt,
                    "matched_by": ["semantic"],
                    "distance": float(distance) if distance is not None else None,
                }
            )
        return hits, unreadable

    @staticmethod
    def _check_embedding_config(collection: Collection, ef: EmbeddingFunction) -> None:
        try:
            persisted = (collection.configuration_json or {}).get("embedding_function")
        except Exception:
            persisted = None
        if not isinstance(persisted, dict):
            raise ReaderError("Index embedding configuration could not be verified")
        if persisted.get("name") != ef.name():
            raise ReaderError("Index embedding configuration does not match the configured embedding")
        stored_config = persisted.get("config")
        try:
            actual_config = ef.get_config()
        except Exception:
            raise ReaderError("Index embedding configuration could not be verified") from None
        if not isinstance(stored_config, dict) or not isinstance(actual_config, dict):
            raise ReaderError("Index embedding configuration could not be verified")
        for key in _EMBEDDING_CONFIG_KEYS:
            if stored_config.get(key) != actual_config.get(key):
                raise ReaderError(
                    "Index embedding configuration does not match the configured embedding"
                )

    def _hybrid_merge(self, lex_hits: list[dict], sem_hits: list[dict], limit: int) -> list[dict]:
        pool_size = min(60, max(10, limit * 3))
        scores: dict[str, float] = {}
        merged: dict[str, dict] = {}
        for hits in (lex_hits[:pool_size], sem_hits[:pool_size]):
            for rank, hit in enumerate(hits, 1):
                doc_id = hit["document_id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)
                if doc_id not in merged:
                    merged[doc_id] = dict(hit, matched_by=list(hit["matched_by"]))
                for method in hit["matched_by"]:
                    if method not in merged[doc_id]["matched_by"]:
                        merged[doc_id]["matched_by"].append(method)
                if merged[doc_id]["excerpt"] is None:
                    merged[doc_id]["excerpt"] = hit["excerpt"]
                if merged[doc_id]["distance"] is None:
                    merged[doc_id]["distance"] = hit["distance"]
        ordered = sorted(scores, key=lambda d: (-scores[d], d))
        return [merged[d] for d in ordered[:limit]]
