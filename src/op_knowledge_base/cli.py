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
