# mdtodb

Index a [pdftomd](../pdftomd) Markdown output directory into a ChromaDB
collection, incrementally — one collection item per source document, with
keywords/person/filetype metadata derived from the path. The Markdown
directory is **never modified**.

Two embedding backends are available, selectable with `--embedding`:

| embedding | how it works                                                          |
|-----------|-----------------------------------------------------------------------|
| `default` | Chroma's built-in ONNX MiniLM (downloads a small model on first run)  |
| `gemini`  | Google Gemini embeddings API (`gemini-embedding-001`, configurable)   |

## Install

```bash
uv sync --group dev                 # core + tests
uv sync --extra gemini              # + google-genai (needs GEMINI_API_KEY)
uv sync --extra all
```

Requires Python 3.10-3.13.

The Gemini key comes from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), or
`--gemini-api-key`. Documents are embedded `--embed-batch-size` at a time
(default 100 per API request; `gemini-embedding-001` reads at most 2,048
tokens per document, so a request stays around 200k tokens), and requests
failing with `429 RESOURCE_EXHAUSTED` or a 5xx are retried with exponential
backoff before the document is reported as failed.

## CLI

`CHROMA` is a local directory for a persistent Chroma database; pass
`--host`/`--port` instead to talk to a Chroma server. Exactly one of the two
is required.

```bash
# What would be (re)indexed? Nothing is written.
mdtodb list  ./docs-md ./chroma
mdtodb sync  ./docs-md ./chroma --dry-run

# Incremental sync: only new / changed / stale documents are re-embedded.
mdtodb sync ./docs-md ./chroma
mdtodb sync ./docs-md ./chroma --embedding gemini --gemini-model <model>
mdtodb sync ./docs-md --host localhost --port 8000 --collection mydocs

# Force specific documents (manifest-relative paths), even if up to date.
mdtodb sync ./docs-md ./chroma --select reports/2024/q3.pdf

# Reindex everything; drop items whose document left the manifest.
mdtodb sync ./docs-md ./chroma --force --prune

# Recompute metadata (keywords, person, filetype) on indexed documents,
# without re-embedding. Also runs automatically after a rules-file change.
mdtodb retag ./docs-md ./chroma

# Poll the manifest and keep the collection in sync.
mdtodb watch ./docs-md ./chroma --interval 30 --prune

# Query; optionally filter on extracted metadata.
mdtodb query ./chroma "assurance habitation" -n 10
mdtodb query ./chroma "relevé" --person Estelle -k allemagne --filetype pdf

# Inspect the metadata a path would get; optionally apply the keyword rules
# file and check content rules against a Markdown file.
mdtodb keywords "personnes/Estelle/papiers/Allemagne/ReleveIntegral_ROY_ESTELLE_2014_26-01-2021.pdf"
mdtodb keywords "docs/avis.pdf" --rules rules.toml --markdown docs-md/docs/avis.md
```

Shared options: `--collection/-c` (default `documents`), `--embedding/-e`,
`--gemini-model`, `--gemini-api-key`, `--stopword/-w` (repeatable, extra
keyword stopwords), `--batch-size` (documents per upsert, default 50),
`--rules/-r` (keyword rules TOML file; also `MDTODB_RULES`).

### Change detection

State lives **in the collection itself**: each item's metadata records the
manifest fingerprint, the sha256 of the indexed Markdown, and the embedding
identifier. A document is (re)indexed when it is `new`, its fingerprint
`changed`, its Markdown `markdown-changed`, the embedding
`embedding-changed`, or it was selected / `forced`. Two further reasons only
refresh metadata (no re-embedding): `rules-changed` (the keyword rules file
changed — or was added/removed — since the document was indexed) and `retag`
(the `mdtodb retag` command). Collection ids are the manifest's relative
source paths.

When pdftomd relocates a document, its manifest entry keeps the same engine
and fingerprint under the new path. mdtodb matches new paths to collection
ids that have left the manifest using these two fields and reports them as
`moved` (`old/path.pdf -> new/path.pdf`). It reuses the stored embedding when
the Markdown hash and embedding identifier are unchanged, while refreshing
`file`, `markdown`, `person`, `filetype`, and all path/content keyword rules.
No manifest migration or full reindex is needed for existing collections.

The new record is written before the old id is deleted; successful moves
remove the old id **even without `--prune`**. If writing the replacement
fails, the old record is kept, including with `--prune`. Chroma does not
provide an atomic rename: a failed deletion can leave both ids in place;
`--prune` on a later sync removes the obsolete one.

Copies whose old path is still in the manifest remain `new`. Duplicate
fingerprints are paired one-to-one in sorted path order. A changed source
fingerprint or conversion engine cannot be identified as a move and falls
back to new/orphan handling. A matched move with changed Markdown, a changed
embedding identifier, or `--force` / `--select` is re-embedded (and reported
with that reason), but still removes its old id after a successful write.
Moves outside `--select`, and moves whose Markdown cannot be read, keep
their old ids even with `--prune`. Only unmatched ids absent from the
manifest are orphans, deleted only with `--prune`.

Note: a Chroma collection is bound to its embedding function at creation —
when switching `--embedding`, use a different `--collection` (or a fresh
database).

## Metadata

| key           | value                                                              |
|---------------|---------------------------------------------------------------------|
| `file`        | source path relative to the documents root (manifest key)           |
| `markdown`    | `.md` path relative to the pdftomd output dir (from the manifest)   |
| `filetype`    | lowercase source extension without dot (`pdf`, `png`, ...)          |
| `person`      | directory after a `personnes` component, if any                     |
| `keywords`    | keyword list extracted from the path (omitted when empty)           |
| `engine`      | conversion engine recorded by pdftomd                               |
| `fingerprint` | content fingerprint recorded by pdftomd                             |
| `md_sha256`   | sha256 of the indexed Markdown text                                 |
| `embedding`   | embedding identifier (`default`, `gemini:<model>`)                  |
| `indexed_at`  | Unix timestamp of the upsert                                        |
| `rules_sha256`| sha256 of the keyword rules file in effect at index time            |

### Keyword rules

Keywords are derived from the document's path (every directory plus the
filename without extension):

* accents stripped, lowercased (`Téléphonie` → `telephonie`);
* split on every non-alphanumeric character;
* a small French + English stopword list is removed (extendable with
  `--stopword`); `personnes` is **not** a stopword;
* tokens shorter than 2 chars are dropped;
* bare numbers are dropped, except 4-digit years in 1900–2099 — including
  years recovered from dates inside a name (`26-01-2021`, `01/2024`,
  `20240315` → `2021`, `2024`, `2024`).

Query-side, keyword filtering uses Chroma's `$contains` on the `keywords`
array: `{"keywords": {"$contains": "estelle"}}`.

### Keyword rules file

A TOML file can add keywords at index time when regexes match the document's
relative path and/or its Markdown content:

```toml
[options]
ignore_case = true          # default true; applies to every rule

[[rule]]
keywords = ["impots", "fiscal"]   # required, non-empty list
path = 'imp[oô]ts|taxe'           # optional, regex searched in the rel POSIX path
content = 'avis\s+d.imp[oô]t'     # optional, regex searched in the Markdown text
ignore_case = false               # optional per-rule override
```

When a rule sets both `path` and `content`, **both** must match (AND); a rule
needs at least one of them. Keywords are normalized like path keywords
(accents stripped, lowercased) and appended after them, deduplicated.
Matching rules apply to every document at index time; the file's sha256 is
recorded as `rules_sha256`, so editing the file marks indexed documents
`rules-changed` and refreshes their metadata on the next `sync` without
re-embedding. `mdtodb retag` forces the same refresh on demand.

The rules file is resolved in this order: `--rules/-r`, the `MDTODB_RULES`
environment variable, then `~/.config/mdtodb/rules.toml` if it exists.
Preview the keywords a document would get with:

```bash
mdtodb keywords "docs/avis d'imposition.pdf" --rules rules.toml --markdown docs-md/docs/avis.md
```

### Person rule

`person` is the directory name immediately following a `personnes`
component (case-insensitive): `personnes/Estelle/papiers/x.pdf` →
`Estelle`. When `personnes` directly introduces the file
(`personnes/x.pdf`), no person is recorded.

## Library

```python
from mdtodb import Indexer, Manifest, open_collection

collection, embedding_name = open_collection("./chroma", name="documents")
indexer = Indexer("docs-md/", collection, embedding_name=embedding_name)
plan = indexer.plan()                          # dry run, reads only .md files
for item in plan.to_index:
    print(item.reason.value, item.rel_path)
result = indexer.execute(plan, prune=True)
# result.moved contains relocated paths (separate from result.indexed).
result = indexer.sync(select=["a/b.pdf"])      # force selected documents

# Single documents / removal.
indexer.index_document("personnes/E/x.pdf", "# Markdown", engine="marker", fingerprint="...")
indexer.remove("personnes/E/x.pdf")

# Disk-less usage: no pdftomd directory needed.
indexer = Indexer(
    None, collection,
    manifest=Manifest.from_entries({"a/b.pdf": {"fingerprint": "...", "engine": "marker"}}),
    read_markdown=lambda rel: get_markdown(rel),
)
```

## Development

```bash
uv run pytest
```
