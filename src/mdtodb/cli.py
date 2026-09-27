from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .embeddings import DEFAULT_GEMINI_MODEL, EMBEDDINGS, get_embedding
from .metadata import metadata_for, normalize_keyword
from .rules import KeywordRules, RulesError, load_rules
from .store import open_collection
from .sync import Indexer, IndexPlan, PlannedItem

app = typer.Typer(
    help="Index a pdftomd Markdown output directory into ChromaDB. One-way, incremental.",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
)

CollectionName = typer.Option("documents", "--collection", "-c", help="Chroma collection name.")
Host = typer.Option(None, "--host", help="Chroma server host (instead of a local CHROMA path).")
Port = typer.Option(8000, "--port", help="Chroma server port.")
Embedding = typer.Option("default", "--embedding", "-e", help=f"Embedding function: {', '.join(EMBEDDINGS)}.", case_sensitive=False)
GeminiModel = typer.Option(DEFAULT_GEMINI_MODEL, "--gemini-model", help="(gemini) Embedding model name.")
GeminiKey = typer.Option(None, "--gemini-api-key", envvar="GEMINI_API_KEY", help="(gemini) API key.", show_default=False)
Stopword = typer.Option(None, "--stopword", "-w", help="Extra keyword stopword. Repeatable.")
BatchSize = typer.Option(50, "--batch-size", min=1, help="Documents upserted per Chroma call.")
Rules = typer.Option(None, "--rules", "-r", envvar="MDTODB_RULES", help="Keyword rules TOML file (default: ~/.config/mdtodb/rules.toml if present).")


def _embedding(name: str, model: str, api_key: Optional[str]):
    """Resolve the embedding function — monkeypatched by tests to stay offline."""
    return get_embedding(name, model_name=model, api_key=api_key)


def _chroma_target(chroma: Optional[Path], host: Optional[str]) -> None:
    if (chroma is None) == (host is None):
        typer.secho("give exactly one of CHROMA (local path) or --host (server)", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _collection(
    chroma: Optional[Path],
    host: Optional[str],
    port: int,
    name: str,
    embedding: str,
    model: str,
    api_key: Optional[str],
):
    _chroma_target(chroma, host)
    try:
        ef, ef_name = _embedding(embedding, model, api_key)
        return open_collection(
            chroma,
            host=host,
            port=port,
            name=name,
            embedding=ef,
            embedding_name=ef_name,
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _rules(path: Optional[Path]) -> Optional[KeywordRules]:
    try:
        return load_rules(path)
    except (RulesError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _indexer(
    md_dir: Path,
    collection,
    ef_name: str,
    stopwords: Optional[list[str]],
    rules: Optional[KeywordRules] = None,
) -> Indexer:
    try:
        return Indexer(md_dir, collection, embedding_name=ef_name, stopwords=stopwords, rules=rules)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _print_plan(plan: IndexPlan, *, verbose: bool) -> None:
    for item in plan.to_index:
        typer.echo(f"[{item.reason.value:>14}] {item.rel_path}")
    if verbose:
        for rel in plan.up_to_date:
            typer.echo(f"[    up-to-date] {rel}")
    for orphan in plan.orphans:
        typer.echo(f"[        orphan] {orphan}")
    for rel, msg in plan.errors.items():
        typer.secho(f"[         error] {rel}: {msg}", fg=typer.colors.RED)
    typer.echo(
        f"-- {len(plan.to_index)} to index, {len(plan.up_to_date)} up to date, "
        f"{len(plan.orphans)} orphan item(s), {len(plan.errors)} error(s)",
        err=True,
    )


def _run(indexer: Indexer, plan: IndexPlan, prune: bool, batch_size: int) -> int:
    def done(item: PlannedItem, i: int, n: int) -> None:
        typer.echo(f"[{i}/{n}] {item.rel_path} ({item.reason.value})", err=True)

    def error(item: PlannedItem, exc: Exception) -> None:
        typer.secho(f"  FAILED {item.rel_path}: {exc}", fg=typer.colors.RED, err=True)

    result = indexer.execute(plan, prune=prune, batch_size=batch_size, on_done=done, on_error=error)
    typer.echo(f"-- indexed {len(result.indexed)}, failed {len(result.failed)}, pruned {len(result.pruned)}", err=True)
    return 1 if result.failed else 0


@app.callback()
def _main(ctx: typer.Context, version: bool = typer.Option(False, "--version", is_eager=True)) -> None:
    if version:
        typer.echo(f"mdtodb {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def sync(
    md_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, help="pdftomd output directory (read-only)."),
    chroma: Optional[Path] = typer.Argument(None, help="Local Chroma storage directory (omit when using --host)."),
    host: Optional[str] = Host,
    port: int = Port,
    collection: str = CollectionName,
    embedding: str = Embedding,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
    stopword: Optional[list[str]] = Stopword,
    rules: Optional[Path] = Rules,
    batch_size: int = BatchSize,
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="List what would be (re)indexed and exit."),
    force: bool = typer.Option(False, "--force", "-f", help="Reindex everything even if up to date."),
    select: Optional[list[str]] = typer.Option(None, "--select", "-s", help="Reindex only these manifest paths; implies force for them. Repeatable."),
    prune: bool = typer.Option(False, "--prune", help="Delete collection items whose document left the manifest."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also list up-to-date documents."),
) -> None:
    """Synchronise MD_DIR into a Chroma collection, reindexing only stale documents."""
    coll, ef_name = _collection(chroma, host, port, collection, embedding, gemini_model, gemini_api_key)
    indexer = _indexer(md_dir, coll, ef_name, stopword, _rules(rules))
    try:
        plan = indexer.plan(force=force, select=select or None)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    _print_plan(plan, verbose=verbose or dry_run)
    if dry_run:
        raise typer.Exit(1 if plan.errors else 0)
    raise typer.Exit(_run(indexer, plan, prune, batch_size))


@app.command("list")
def list_cmd(
    md_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    chroma: Optional[Path] = typer.Argument(None),
    host: Optional[str] = Host,
    port: int = Port,
    collection: str = CollectionName,
    embedding: str = Embedding,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
    stopword: Optional[list[str]] = Stopword,
    rules: Optional[Path] = Rules,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also list up-to-date documents."),
) -> None:
    """List the documents that would be (re)indexed, without writing anything."""
    coll, ef_name = _collection(chroma, host, port, collection, embedding, gemini_model, gemini_api_key)
    plan = _indexer(md_dir, coll, ef_name, stopword, _rules(rules)).plan()
    _print_plan(plan, verbose=verbose)
    raise typer.Exit(1 if plan.errors else 0)


@app.command()
def watch(
    md_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    chroma: Optional[Path] = typer.Argument(None),
    host: Optional[str] = Host,
    port: int = Port,
    collection: str = CollectionName,
    embedding: str = Embedding,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
    stopword: Optional[list[str]] = Stopword,
    rules: Optional[Path] = Rules,
    batch_size: int = BatchSize,
    interval: float = typer.Option(30.0, "--interval", "-i", help="Seconds between manifest reloads."),
    prune: bool = typer.Option(False, "--prune", help="Delete collection items whose document left the manifest."),
) -> None:
    """Keep the collection in sync with MD_DIR, re-reading the manifest periodically."""
    coll, ef_name = _collection(chroma, host, port, collection, embedding, gemini_model, gemini_api_key)
    indexer = _indexer(md_dir, coll, ef_name, stopword, _rules(rules))
    typer.echo(f"watching {md_dir} every {interval:g}s (Ctrl-C to stop)", err=True)
    try:
        while True:
            indexer.reload()
            plan = indexer.plan()
            if plan.to_index or plan.errors or (prune and plan.orphans):
                _run(indexer, plan, prune, batch_size)
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("stopped", err=True)


@app.command()
def query(
    target: str = typer.Argument(..., help="Query text, or CHROMA path when TEXT is also given."),
    text: Optional[str] = typer.Argument(None, help="Query text (CHROMA given as first argument)."),
    host: Optional[str] = Host,
    port: int = Port,
    collection: str = CollectionName,
    embedding: str = Embedding,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
    results: int = typer.Option(5, "--results", "-n", min=1, help="Number of hits."),
    person: Optional[str] = typer.Option(None, "--person", help="Restrict to this person directory."),
    keyword: Optional[list[str]] = typer.Option(None, "--keyword", "-k", help="Restrict to documents containing this keyword. Repeatable."),
    filetype: Optional[str] = typer.Option(None, "--filetype", help="Restrict to this extension (e.g. pdf)."),
) -> None:
    """Query the collection and print the top hits.

    Usage: ``mdtodb query TEXT --host ...`` or ``mdtodb query CHROMA TEXT``.
    """
    chroma = Path(target) if text is not None else None
    if text is None:
        text = target
    coll, _ef_name = _collection(chroma, host, port, collection, embedding, gemini_model, gemini_api_key)
    clauses = []
    if person:
        clauses.append({"person": person})
    for k in keyword or ():
        clauses.append({"keywords": {"$contains": normalize_keyword(k)}})
    if filetype:
        clauses.append({"filetype": filetype.lower().lstrip(".")})
    where = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)
    res = coll.query(query_texts=[text], n_results=results, where=where, include=["metadatas", "documents", "distances"])
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    docs = res["documents"][0]
    dists = res["distances"][0]
    for rank, (_id, meta, doc, dist) in enumerate(zip(ids, metas, docs, dists), 1):
        typer.echo(f"{rank}. {dist:.4f}  {_id}")
        if meta.get("person"):
            typer.echo(f"   person: {meta['person']}")
        if meta.get("keywords"):
            typer.echo(f"   keywords: {', '.join(meta['keywords'])}")
        snippet = " ".join((doc or "").split())[:200]
        if snippet:
            typer.echo(f"   {snippet}")


@app.command()
@app.command()
def retag(
    md_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    chroma: Optional[Path] = typer.Argument(None),
    host: Optional[str] = Host,
    port: int = Port,
    collection: str = CollectionName,
    embedding: str = Embedding,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
    stopword: Optional[list[str]] = Stopword,
    rules: Optional[Path] = Rules,
    batch_size: int = BatchSize,
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="List what would be re-tagged and exit."),
) -> None:
    """Recompute metadata (keywords, person, filetype) for every indexed document without re-embedding."""
    coll, ef_name = _collection(chroma, host, port, collection, embedding, gemini_model, gemini_api_key)
    indexer = _indexer(md_dir, coll, ef_name, stopword, _rules(rules))
    plan = indexer.plan(retag=True)
    _print_plan(plan, verbose=dry_run)
    if dry_run:
        raise typer.Exit(1 if plan.errors else 0)
    raise typer.Exit(_run(indexer, plan, False, batch_size))


@app.command()
def keywords(
    path: str = typer.Argument(..., help="Document path to analyse (relative POSIX style)."),
    stopword: Optional[list[str]] = Stopword,
    rules: Optional[Path] = Rules,
    markdown: Optional[Path] = typer.Option(None, "--markdown", exists=True, dir_okay=False, help="Markdown file whose text content rules are checked against."),
) -> None:
    """Print the metadata mdtodb would store for PATH, as JSON."""
    text = markdown.read_text("utf-8") if markdown is not None else None
    meta = metadata_for(path, stopwords=stopword, text=text, rules=_rules(rules))
    typer.echo(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    app()
