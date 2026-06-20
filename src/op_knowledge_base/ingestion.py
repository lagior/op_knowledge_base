"""Orchestrates the ingestion pipeline: detect changes -> chunk -> store."""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from op_knowledge_base.change_detection import content_hash, save_state
from op_knowledge_base.config import load_config
from op_knowledge_base.models import IngestionResult
from op_knowledge_base.sources import confluence as confluence_source
from op_knowledge_base.sources import git as git_source
from op_knowledge_base.store import (
    add_documents,
    delete_by_doc_id,
    delete_by_ids,
    get_chunk_hashes,
    init_store,
)

logger = logging.getLogger(__name__)


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
    Handles errors gracefully: logs failures and continues processing.
    """
    result = IngestionResult(source_type=source_type)

    try:
        changed_docs, deleted_ids, new_state = fetch_fn(config)
    except Exception as e:
        msg = f"Failed to fetch {source_type} documents: {e}"
        logger.error(msg)
        result.errors.append(msg)
        return result

    store = init_store(config)
    splitter = _build_splitter(config)

    for doc_id in deleted_ids:
        try:
            delete_by_doc_id(store, doc_id)
            result.documents_deleted += 1
        except Exception as e:
            msg = f"Failed to delete {doc_id}: {e}"
            logger.error(msg)
            result.errors.append(msg)

    if not changed_docs:
        save_state(source_type, new_state)
        return result

    chunks = splitter.split_documents(changed_docs)

    # Compute chunk hashes
    for chunk in chunks:
        chunk.metadata["chunk_hash"] = content_hash(chunk.page_content)

    # Deduplicate per document
    updated_doc_ids = {doc.metadata["doc_id"] for doc in changed_docs}
    new_chunks_to_add = []
    for doc_id in updated_doc_ids:
        try:
            old_hashes = get_chunk_hashes(store, doc_id)
        except Exception as e:
            msg = f"Failed to get chunk hashes for {doc_id}: {e}"
            logger.error(msg)
            result.errors.append(msg)
            # Fall back to adding all chunks for this doc (no dedup)
            new_chunks_to_add.extend(
                c for c in chunks if c.metadata["doc_id"] == doc_id
            )
            continue

        doc_chunks = [c for c in chunks if c.metadata["doc_id"] == doc_id]
        new_hash_set = {c.metadata["chunk_hash"] for c in doc_chunks}

        for chunk in doc_chunks:
            if chunk.metadata["chunk_hash"] in old_hashes:
                result.chunks_skipped += 1
            else:
                new_chunks_to_add.append(chunk)

        ids_to_delete = []
        for old_hash, chroma_ids in old_hashes.items():
            if old_hash not in new_hash_set:
                ids_to_delete.extend(chroma_ids)
        delete_by_ids(store, ids_to_delete)

    if new_chunks_to_add:
        try:
            add_documents(store, new_chunks_to_add)
        except Exception as e:
            msg = f"Failed to store chunks: {e}"
            logger.error(msg)
            result.errors.append(msg)
            return result

    result.documents_processed = len(changed_docs)
    save_state(source_type, new_state)

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
    """Run ingestion for all configured sources. Continues on failure."""
    if config is None:
        config = load_config()
    results = []
    for ingest_fn in [ingest_confluence, ingest_git]:
        results.append(ingest_fn(config))
    return results
