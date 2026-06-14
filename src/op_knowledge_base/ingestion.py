"""Orchestrates the ingestion pipeline: detect changes -> chunk -> store."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from op_knowledge_base.config import load_config
from op_knowledge_base.models import IngestionResult
from op_knowledge_base.sources.confluence import fetch_changed_documents
from op_knowledge_base.store import add_documents, delete_by_doc_id, init_store


def _build_splitter(config: dict) -> RecursiveCharacterTextSplitter:
    """Create a text splitter from config."""
    chunking = config["chunking"]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunking["chunk_size"],
        chunk_overlap=chunking["chunk_overlap"],
    )


def ingest_confluence(config: dict | None = None) -> IngestionResult:
    """Run the full Confluence ingestion pipeline.

    Detects changed pages, chunks them, stores embeddings,
    and removes deleted pages from the store.
    """
    if config is None:
        config = load_config()

    result = IngestionResult(source_type="confluence")

    # Detect changes
    changed_docs, deleted_ids = fetch_changed_documents(config)

    # Initialize store and splitter
    store = init_store(config)
    splitter = _build_splitter(config)

    # Handle deletions
    for doc_id in deleted_ids:
        delete_by_doc_id(store, doc_id)
        result.documents_deleted += 1

    if not changed_docs:
        return result

    # Split changed documents into chunks, preserving metadata
    chunks = splitter.split_documents(changed_docs)

    # Delete old chunks for changed documents before re-adding
    updated_doc_ids = {doc.metadata["doc_id"] for doc in changed_docs}
    for doc_id in updated_doc_ids:
        delete_by_doc_id(store, doc_id)

    # Store new chunks
    add_documents(store, chunks)
    result.documents_processed = len(changed_docs)

    return result
