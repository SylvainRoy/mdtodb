"""One-way, incremental synchronisation of a pdftomd output dir into ChromaDB.

The Markdown directory is only ever read. State lives inside the Chroma
collection itself (each item's metadata records the source fingerprint, the
Markdown sha, and the embedding identifier). A document is (re)indexed when
any of the following holds:

* it is not in the collection,
* its manifest fingerprint differs from the recorded one,
* the Markdown content sha differs from the recorded ``md_sha256``,
* the recorded embedding identifier differs from the one in use,
* it was explicitly selected / ``force`` was requested.

Collection ids are the manifest's relative source paths. Ids present in the
collection but absent from the manifest are orphans, deleted only with
``prune``.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from chromadb.api.models.Collection import Collection

from .manifest import MANIFEST_NAME, Manifest, markdown_path_for
from .metadata import metadata_for

_BATCH_GET = 1000


class Reason(str, Enum):
    NEW = "new"
    CHANGED = "changed"
    MARKDOWN_CHANGED = "markdown-changed"
    EMBEDDING_CHANGED = "embedding-changed"
    FORCED = "forced"


@dataclass(frozen=True)
class PlannedItem:
    rel_path: str
    markdown: Path | None  # absolute path of the .md to index, None in disk-less mode
    reason: Reason


@dataclass
class IndexPlan:
    to_index: list[PlannedItem] = field(default_factory=list)
    up_to_date: list[str] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)  # collection ids without a manifest entry
    errors: dict[str, str] = field(default_factory=dict)  # entries whose state could not be determined


@dataclass
class IndexResult:
    indexed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    pruned: list[str] = field(default_factory=list)


def md_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Indexer:
    def __init__(
        self,
        md_dir: str | Path | None,
        collection: Collection,
        *,
        manifest: Manifest | None = None,
        embedding_name: str = "default",
        read_markdown: Callable[[str], str] | None = None,
        stopwords: Iterable[str] | None = None,
    ) -> None:
        self.md_dir = Path(md_dir) if md_dir is not None else None
        self.collection = collection
        self.embedding_name = embedding_name
        self.stopwords = stopwords
        if read_markdown is None:
            if self.md_dir is None:
                raise ValueError("md_dir is required when no read_markdown callable is given")
            read_markdown = self._read_from_disk
        self.read_markdown = read_markdown
        if manifest is None:
            if self.md_dir is None:
                raise ValueError("md_dir is required when no manifest is given")
            self.manifest = self._load_manifest()
        else:
            self.manifest = manifest

    def _load_manifest(self) -> Manifest:
        path = self.md_dir / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(f"no pdftomd manifest at {path} — run `pdftomd sync` first")
        return Manifest.load(path)

    def reload(self) -> None:
        """Re-read the manifest from disk (for ``watch`` loops)."""
        self.manifest = self._load_manifest()

    def _read_from_disk(self, rel_path: str) -> str:
        entry = self.manifest.entries[rel_path]
        output = entry.output or markdown_path_for(rel_path)
        return (self.md_dir / output).read_text("utf-8")

    def _markdown_path(self, rel_path: str) -> Path | None:
        if self.md_dir is None:
            return None
        entry = self.manifest.entries[rel_path]
        return self.md_dir / (entry.output or markdown_path_for(rel_path))

    # -- planning ------------------------------------------------------------

    def _rel(self, selection: str | Path) -> str:
        p = Path(selection)
        if p.is_absolute() and self.md_dir is not None:
            try:
                p = p.resolve().relative_to(self.md_dir)
            except ValueError:
                pass
        return PurePosixPath(p.as_posix()).as_posix()

    def _collection_state(self) -> dict[str, dict]:
        """All collection ids mapped to their metadata, paged in batches."""
        state: dict[str, dict] = {}
        offset = 0
        while True:
            page = self.collection.get(limit=_BATCH_GET, offset=offset, include=["metadatas"])
            for _id, meta in zip(page["ids"], page["metadatas"] or []):
                state[_id] = meta or {}
            if len(page["ids"]) < _BATCH_GET:
                break
            offset += _BATCH_GET
        return state

    def plan(self, *, force: bool = False, select: Iterable[str | Path] | None = None) -> IndexPlan:
        """Compute what would be (re)indexed. Never writes to the collection.

        ``select`` restricts the plan to the given manifest paths and forces
        their reindexing; a selection missing from the manifest is a
        ``FileNotFoundError``.
        """
        selected = {self._rel(s) for s in select} if select else None
        plan = IndexPlan()
        state = self._collection_state()

        for rel, entry in self.manifest.entries.items():
            if selected is not None and rel not in selected:
                continue
            try:
                markdown = self.read_markdown(rel)
            except Exception as exc:
                plan.errors[rel] = str(exc)
                continue
            existing = state.get(rel)
            if existing is None:
                reason = Reason.NEW
            elif selected is not None or force:
                reason = Reason.FORCED
            elif existing.get("fingerprint") != entry.fingerprint:
                reason = Reason.CHANGED
            elif existing.get("md_sha256") != md_sha256(markdown):
                reason = Reason.MARKDOWN_CHANGED
            elif existing.get("embedding") != self.embedding_name:
                reason = Reason.EMBEDDING_CHANGED
            else:
                plan.up_to_date.append(rel)
                continue
            plan.to_index.append(PlannedItem(rel, self._markdown_path(rel), reason))

        if selected is not None:
            missing = selected - set(self.manifest.entries)
            if missing:
                raise FileNotFoundError(f"Selected files not found in manifest: {sorted(missing)}")

        plan.orphans = sorted(set(state) - set(self.manifest.entries))
        return plan

    # -- execution -----------------------------------------------------------

    def _item_metadata(self, item: PlannedItem, markdown: str) -> dict:
        entry = self.manifest.entries[item.rel_path]
        return metadata_for(
            item.rel_path,
            markdown=entry.output or markdown_path_for(item.rel_path),
            engine=entry.engine,
            fingerprint=entry.fingerprint,
            md_sha256=md_sha256(markdown),
            embedding=self.embedding_name,
            indexed_at=time.time(),
            stopwords=self.stopwords,
        )

    def execute(
        self,
        plan: IndexPlan,
        *,
        prune: bool = False,
        batch_size: int = 50,
        on_progress: Callable[[PlannedItem, int, int], None] | None = None,
        on_done: Callable[[PlannedItem, int, int], None] | None = None,
        on_error: Callable[[PlannedItem, Exception], None] | None = None,
    ) -> IndexResult:
        result = IndexResult()
        result.failed.update(plan.errors)
        total = len(plan.to_index)

        batch: list[tuple[PlannedItem, str, dict, int]] = []  # item, document, metadata, index

        def flush() -> None:
            if not batch:
                return
            try:
                self.collection.upsert(
                    ids=[i.rel_path for i, *_ in batch],
                    documents=[doc for _, doc, *_ in batch],
                    metadatas=[meta for *_, meta, _ in batch],
                )
            except Exception as exc:
                for item, _doc, _meta, _i in batch:
                    result.failed[item.rel_path] = str(exc)
                    if on_error:
                        on_error(item, exc)
            else:
                for item, _doc, _meta, i in batch:
                    result.indexed.append(item.rel_path)
                    if on_done:
                        on_done(item, i, total)
            batch.clear()

        for index, item in enumerate(plan.to_index, 1):
            if on_progress:
                on_progress(item, index, total)
            try:
                markdown = self.read_markdown(item.rel_path)
                if not markdown:
                    raise ValueError("empty markdown")
                meta = self._item_metadata(item, markdown)
            except Exception as exc:
                result.failed[item.rel_path] = str(exc)
                if on_error:
                    on_error(item, exc)
                continue
            batch.append((item, markdown, meta, index))
            if len(batch) >= batch_size:
                flush()
        flush()

        if prune and plan.orphans:
            for offset in range(0, len(plan.orphans), batch_size):
                chunk = plan.orphans[offset : offset + batch_size]
                self.collection.delete(ids=chunk)
                result.pruned.extend(chunk)
        return result

    def index_document(
        self,
        rel_path: str,
        markdown: str,
        *,
        engine: str | None = None,
        fingerprint: str | None = None,
        output: str | None = None,
    ) -> dict:
        """Index a single document under ``rel_path``; returns the stored metadata."""
        if not markdown:
            raise ValueError("empty markdown")
        meta = metadata_for(
            rel_path,
            markdown=output if output is not None else markdown_path_for(rel_path),
            engine=engine,
            fingerprint=fingerprint,
            md_sha256=md_sha256(markdown),
            embedding=self.embedding_name,
            indexed_at=time.time(),
            stopwords=self.stopwords,
        )
        self.collection.upsert(ids=[rel_path], documents=[markdown], metadatas=[meta])
        return meta

    def remove(self, rel_path: str) -> None:
        self.collection.delete(ids=[rel_path])

    def sync(
        self,
        *,
        force: bool = False,
        select: Iterable[str | Path] | None = None,
        prune: bool = False,
        batch_size: int = 50,
        **callbacks,
    ) -> IndexResult:
        return self.execute(
            self.plan(force=force, select=select), prune=prune, batch_size=batch_size, **callbacks
        )
