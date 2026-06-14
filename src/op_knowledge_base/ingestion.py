"""Orchestrates the ingestion pipeline: detect changes -> chunk -> store."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from op_knowledge_base.config import load_config
from op_knowledge_base.models import IngestionResult
from op_knowledge_base.sources import confluence as confluence_source
from op_knowledge_base.sources import git as git_source
from op_knowledge_base.store import add_documents, delete_by_doc_id, init_store


def _build_splitter(config: dict) -> RecursiveCharacterTextSplitter:
    """Create a text splitter from config."""
    chunking = config["chunking"]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunking["chunk_size"],
        chunk_overlap=chunking["chunk_overlap"],
    )


def _ingest_source(source_type: str, fetch_fn, config: dict) -> IngestionResult:
    """Run the ingestion pipeline for a given source.

    Detects changed documents, chunks them, stores embeddings,
    and removes deleted documents from the store.
    """
    result = IngestionResult(source_type=source_type)

    changed_docs, deleted_ids = fetch_fn(config)

    store = init_store(config)
    splitter = _build_splitter(config)

    for doc_id in deleted_ids:
        delete_by_doc_id(store, doc_id)
        result.documents_deleted += 1

    if not changed_docs:
        return result

    chunks = splitter.split_documents(changed_docs)

    updated_doc_ids = {doc.metadata["doc_id"] for doc in changed_docs}
    for doc_id in updated_doc_ids:
        delete_by_doc_id(store, doc_id)

    add_documents(store, chunks)
    result.documents_processed = len(changed_docs)

    return result


def ingest_confluence(config: dict | None = None) -> IngestionResult:
    """Run the full Confluence ingestion pipeline."""
    if config is None:
        config = load_config()
    return _ingest_source("confluence", confluence_source.fetch_changed_documents, config)


def ingest_git(config: dict | None = None) -> IngestionResult:
    """Run the full Git ingestion pipeline."""
    if config is None:
        config = load_config()
    return _ingest_source("git", git_source.fetch_changed_documents, config)


def ingest_all(config: dict | None = None) -> list[IngestionResult]:
    """Run ingestion for all configured sources."""
    if config is None:
        config = load_config()
    return [ingest_confluence(config), ingest_git(config)]
