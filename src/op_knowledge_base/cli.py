"""CLI entry points for the RAG system."""

import click

from op_knowledge_base.config import load_config


@click.group()
def main():
    """RAG system for evolving knowledge sources."""


@main.group()
def ingest():
    """Ingest documents from sources."""


def _echo_result(result):
    """Print an ingestion result."""
    click.echo(
        f"  [{result.source_type}] {result.documents_processed} updated, "
        f"{result.documents_deleted} deleted"
    )
    for err in result.errors:
        click.echo(f"  Error: {err}")


@ingest.command()
def confluence():
    """Ingest changed pages from Confluence."""
    from op_knowledge_base.ingestion import ingest_confluence

    config = load_config()
    click.echo("Ingesting from Confluence...")
    _echo_result(ingest_confluence(config))


@ingest.command()
def git():
    """Ingest changed files from Git repositories."""
    from op_knowledge_base.ingestion import ingest_git

    config = load_config()
    click.echo("Ingesting from Git repos...")
    _echo_result(ingest_git(config))


@ingest.command(name="all")
def ingest_all_cmd():
    """Ingest from all configured sources."""
    from op_knowledge_base.ingestion import ingest_all

    config = load_config()
    click.echo("Ingesting from all sources...")
    for result in ingest_all(config):
        _echo_result(result)
    click.echo("Done.")


@main.command()
@click.argument("question")
@click.option("--top-k", default=5, help="Number of chunks to retrieve.")
@click.option("--source", type=click.Choice(["confluence", "git"]), default=None,
              help="Filter results to a specific source type.")
def query(question, top_k, source):
    """Ask a question against the knowledge base."""
    from op_knowledge_base.query import ask

    config = load_config()
    result = ask(config, question, top_k=top_k, source_type=source)

    click.echo(f"\n{result['answer']}\n")

    if result["sources"]:
        click.echo("Sources:")
        for src in result["sources"]:
            line = f"  - [{src['source_type']}] {src['title']}"
            if src.get("last_updated"):
                line += f" (updated: {src['last_updated']})"
            click.echo(line)
