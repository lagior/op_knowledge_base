"""CLI entry points for the RAG system."""

import click

from op_knowledge_base.config import load_config


@click.group()
def main():
    """RAG system for evolving knowledge sources."""


@main.group()
def ingest():
    """Ingest documents from sources."""


@ingest.command()
def confluence():
    """Ingest changed pages from Confluence."""
    from op_knowledge_base.ingestion import ingest_confluence

    config = load_config()
    click.echo("Ingesting from Confluence...")

    result = ingest_confluence(config)

    click.echo(
        f"Done: {result.documents_processed} updated, "
        f"{result.documents_deleted} deleted"
    )
    if result.errors:
        for err in result.errors:
            click.echo(f"  Error: {err}")
