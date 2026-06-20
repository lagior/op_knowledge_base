"""Tests for the ingestion pipeline."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from op_knowledge_base.ingestion import ingest_confluence, ingest_git


def _make_doc(doc_id, content="test content", source_type="confluence", title="Test Page"):
    """Create a Document with standard metadata."""
    return Document(
        page_content=content,
        metadata={"doc_id": doc_id, "source_type": source_type, "title": title},
    )


def _fake_config():
    return {
        "openai_api_key": "test-key",
        "embedding": {"model": "text-embedding-3-small"},
        "chroma": {
            "persist_directory": "/tmp/test_chroma",
            "collection_name": "test",
        },
        "chunking": {"chunk_size": 1000, "chunk_overlap": 200},
        "confluence": {
            "url": "https://test.atlassian.net",
            "username": "test",
            "api_token": "test",
            "space_key": "TEST",
        },
        "git": {
            "repos": [{"path": "/tmp/test-repo", "extensions": [".md"]}],
        },
    }


# --- Confluence ingestion tests ---


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.confluence.fetch_changed_documents")
def test_ingest_confluence_no_changes(mock_fetch, mock_init_store, mock_save):
    """When nothing changed, no documents are processed."""
    mock_fetch.return_value = ([], [], {})
    mock_init_store.return_value = MagicMock()

    result = ingest_confluence(config=_fake_config())

    assert result.documents_processed == 0
    assert result.documents_deleted == 0


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.confluence.fetch_changed_documents")
def test_ingest_confluence_changed_docs(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Changed documents are chunked and stored."""
    docs = [_make_doc("confluence:1"), _make_doc("confluence:2")]
    mock_fetch.return_value = (docs, [], {})
    mock_init_store.return_value = MagicMock()
    mock_get_hashes.return_value = {}  # No existing chunks

    result = ingest_confluence(config=_fake_config())

    assert result.documents_processed == 2
    assert result.documents_deleted == 0
    assert mock_add.called


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.confluence.fetch_changed_documents")
def test_ingest_confluence_deletions(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Deleted pages are removed from the store."""
    mock_fetch.return_value = ([], ["confluence:99"], {})
    mock_init_store.return_value = MagicMock()

    result = ingest_confluence(config=_fake_config())

    assert result.documents_processed == 0
    assert result.documents_deleted == 1
    mock_delete.assert_called_once()


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.confluence.fetch_changed_documents")
def test_ingest_confluence_mixed(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Handles both changes and deletions in one run."""
    docs = [_make_doc("confluence:1")]
    mock_fetch.return_value = (docs, ["confluence:5", "confluence:6"], {})
    mock_init_store.return_value = MagicMock()
    mock_get_hashes.return_value = {}  # No existing chunks

    result = ingest_confluence(config=_fake_config())

    assert result.documents_processed == 1
    assert result.documents_deleted == 2
    # 2 deletions via delete_by_doc_id
    assert mock_delete.call_count == 2


# --- Git ingestion tests ---


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.git.fetch_changed_documents")
def test_ingest_git_no_changes(mock_fetch, mock_init_store, mock_save):
    """When no git files changed, no documents are processed."""
    mock_fetch.return_value = ([], [], {})
    mock_init_store.return_value = MagicMock()

    result = ingest_git(config=_fake_config())

    assert result.source_type == "git"
    assert result.documents_processed == 0
    assert result.documents_deleted == 0


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.git.fetch_changed_documents")
def test_ingest_git_changed_docs(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Changed git files are chunked and stored."""
    docs = [
        _make_doc("git:repo:file1.md", source_type="git"),
        _make_doc("git:repo:file2.md", source_type="git"),
    ]
    mock_fetch.return_value = (docs, [], {})
    mock_init_store.return_value = MagicMock()
    mock_get_hashes.return_value = {}  # No existing chunks

    result = ingest_git(config=_fake_config())

    assert result.source_type == "git"
    assert result.documents_processed == 2
    assert mock_add.called


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.git.fetch_changed_documents")
def test_ingest_git_deletions(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Deleted git files are removed from the store."""
    mock_fetch.return_value = ([], ["git:repo:old.md"], {})
    mock_init_store.return_value = MagicMock()

    result = ingest_git(config=_fake_config())

    assert result.documents_processed == 0
    assert result.documents_deleted == 1
    mock_delete.assert_called_once()


# --- Chunk-level deduplication tests ---


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.git.fetch_changed_documents")
def test_unchanged_chunks_are_skipped(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Chunks whose content hash already exists in the store are not re-added."""
    from op_knowledge_base.change_detection import content_hash

    doc = _make_doc("git:repo:file.md", content="unchanged content", source_type="git")
    mock_fetch.return_value = ([doc], [], {})
    mock_init_store.return_value = MagicMock()

    # Simulate: store already has a chunk with same hash
    existing_hash = content_hash("unchanged content")
    mock_get_hashes.return_value = {existing_hash: ["existing-id-1"]}

    result = ingest_git(config=_fake_config())

    assert result.chunks_skipped == 1
    assert result.documents_processed == 1
    # No new chunks to add
    mock_add.assert_not_called()
    # No old chunks to delete (hash still present)
    mock_delete_ids.assert_called_once_with(mock_init_store.return_value, [])


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.git.fetch_changed_documents")
def test_changed_chunks_are_added(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Chunks with new content are added to the store."""
    from op_knowledge_base.change_detection import content_hash

    doc = _make_doc("git:repo:file.md", content="new content", source_type="git")
    mock_fetch.return_value = ([doc], [], {})
    mock_init_store.return_value = MagicMock()

    # Store has a chunk with a different hash
    mock_get_hashes.return_value = {content_hash("old content"): ["old-id-1"]}

    result = ingest_git(config=_fake_config())

    assert result.chunks_skipped == 0
    assert result.documents_processed == 1
    # New chunk should be added
    mock_add.assert_called_once()
    # Old chunk should be deleted
    mock_delete_ids.assert_called_once_with(mock_init_store.return_value, ["old-id-1"])


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.git.fetch_changed_documents")
def test_removed_chunks_are_deleted(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Old chunks not present in new version are deleted from the store."""
    from op_knowledge_base.change_detection import content_hash

    # Document now has only "kept content"
    doc = _make_doc("git:repo:file.md", content="kept content", source_type="git")
    mock_fetch.return_value = ([doc], [], {})
    mock_init_store.return_value = MagicMock()

    kept_hash = content_hash("kept content")
    removed_hash = content_hash("removed content")
    mock_get_hashes.return_value = {
        kept_hash: ["keep-id"],
        removed_hash: ["remove-id"],
    }

    result = ingest_git(config=_fake_config())

    # "kept content" chunk is skipped (already exists)
    assert result.chunks_skipped == 1
    # "removed content" chunk should be deleted
    mock_delete_ids.assert_called_once_with(mock_init_store.return_value, ["remove-id"])


# --- Error handling tests ---


@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.confluence.fetch_changed_documents")
def test_ingest_fetch_failure_returns_error(mock_fetch, mock_init_store):
    """Source fetch failure is captured in result.errors."""
    mock_fetch.side_effect = ConnectionError("Confluence is down")

    result = ingest_confluence(config=_fake_config())

    assert result.documents_processed == 0
    assert len(result.errors) == 1
    assert "Confluence is down" in result.errors[0]


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.git.fetch_changed_documents")
def test_ingest_embed_failure_returns_error(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Embedding failure is captured in result.errors and state is NOT saved."""
    docs = [_make_doc("git:repo:file.md", source_type="git")]
    mock_fetch.return_value = (docs, [], {})
    mock_init_store.return_value = MagicMock()
    mock_get_hashes.return_value = {}
    mock_add.side_effect = RuntimeError("Rate limit exceeded")

    result = ingest_git(config=_fake_config())

    assert len(result.errors) == 1
    assert "Rate limit" in result.errors[0]
    # State should NOT be saved on embed failure
    mock_save.assert_not_called()


@patch("op_knowledge_base.ingestion.save_state")
@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_ids")
@patch("op_knowledge_base.ingestion.get_chunk_hashes")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.sources.confluence.fetch_changed_documents")
def test_ingest_delete_failure_continues(
    mock_fetch, mock_init_store, mock_delete, mock_get_hashes, mock_delete_ids, mock_add, mock_save
):
    """Delete failure for one doc doesn't stop processing others."""
    mock_fetch.return_value = ([], ["conf:1", "conf:2"], {})
    mock_init_store.return_value = MagicMock()
    mock_delete.side_effect = [RuntimeError("delete failed"), None]

    result = ingest_confluence(config=_fake_config())

    # First delete failed, second succeeded
    assert result.documents_deleted == 1
    assert len(result.errors) == 1
    assert "conf:1" in result.errors[0]
