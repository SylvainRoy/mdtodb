"""mdtodb: index a pdftomd Markdown output directory into ChromaDB.

Library usage::

    from mdtodb import Indexer, open_collection

    collection, embedding_name = open_collection("chroma/", name="documents")
    indexer = Indexer("docs-md/", collection, embedding_name=embedding_name)
    plan = indexer.plan()            # dry run
    indexer.execute(plan)            # upsert only what is stale

Disk-less usage (no pdftomd directory needed)::

    indexer = Indexer(None, collection,
                      manifest=Manifest.from_entries({"a/b.pdf": {"fingerprint": "...", "engine": "marker"}}),
                      read_markdown=lambda rel: get_text(rel))
"""

from __future__ import annotations

from .embeddings import EMBEDDINGS, get_embedding
from .manifest import MANIFEST_NAME, Manifest, ManifestEntry, markdown_path_for
from .metadata import (
    DEFAULT_STOPWORDS,
    filetype_for,
    keywords_for,
    metadata_for,
    normalize_keyword,
    person_for,
)
from .reader import CorpusReader, ReaderError
from .rules import KeywordRules, load_rules
from .store import open_collection, open_existing_collection
from .sync import Indexer, IndexPlan, IndexResult, PlannedItem, Reason

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_STOPWORDS",
    "EMBEDDINGS",
    "MANIFEST_NAME",
    "CorpusReader",
    "Indexer",
    "IndexPlan",
    "IndexResult",
    "KeywordRules",
    "Manifest",
    "ManifestEntry",
    "PlannedItem",
    "ReaderError",
    "Reason",
    "__version__",
    "filetype_for",
    "get_embedding",
    "keywords_for",
    "load_rules",
    "markdown_path_for",
    "metadata_for",
    "normalize_keyword",
    "open_collection",
    "open_existing_collection",
    "person_for",
]
