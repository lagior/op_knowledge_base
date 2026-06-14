"""Tests for the ingestion pipeline."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from op_knowledge_base.ingestion import ingest_confluence


def _make_doc(doc_id, content="test content", title="Test Page"):
    """Create a Document with standard metadata."""
    return Document(
        page_content=content,
        metadata={"doc_id": doc_id, "source_type": "confluence", "title": title},
    )


@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.ingestion.fetch_changed_documents")
def test_ingest_no_changes(mock_fetch, mock_init_store):
    """When nothing changed, no documents are processed."""
    mock_fetch.return_value = ([], [])
    mock_init_store.return_value = MagicMock()

    result = ingest_confluence(config=_fake_config())

    assert result.documents_processed == 0
    assert result.documents_deleted == 0


@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.ingestion.fetch_changed_documents")
def test_ingest_changed_docs(mock_fetch, mock_init_store, mock_delete, mock_add):
    """Changed documents are chunked and stored."""
    docs = [_make_doc("confluence:1"), _make_doc("confluence:2")]
    mock_fetch.return_value = (docs, [])
    mock_init_store.return_value = MagicMock()

    result = ingest_confluence(config=_fake_config())

    assert result.documents_processed == 2
    assert result.documents_deleted == 0
    assert mock_add.called
    # Old chunks should be deleted before re-adding
    assert mock_delete.call_count == 2


@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.ingestion.fetch_changed_documents")
def test_ingest_deletions(mock_fetch, mock_init_store, mock_delete, mock_add):
    """Deleted pages are removed from the store."""
    mock_fetch.return_value = ([], ["confluence:99"])
    mock_init_store.return_value = MagicMock()

    result = ingest_confluence(config=_fake_config())

    assert result.documents_processed == 0
    assert result.documents_deleted == 1
    mock_delete.assert_called_once()


@patch("op_knowledge_base.ingestion.add_documents")
@patch("op_knowledge_base.ingestion.delete_by_doc_id")
@patch("op_knowledge_base.ingestion.init_store")
@patch("op_knowledge_base.ingestion.fetch_changed_documents")
def test_ingest_mixed(mock_fetch, mock_init_store, mock_delete, mock_add):
    """Handles both changes and deletions in one run."""
    docs = [_make_doc("confluence:1")]
    mock_fetch.return_value = (docs, ["confluence:5", "confluence:6"])
    mock_init_store.return_value = MagicMock()

    result = ingest_confluence(config=_fake_config())

    assert result.documents_processed == 1
    assert result.documents_deleted == 2
    # 2 deletions + 1 old-chunk cleanup = 3 delete calls
    assert mock_delete.call_count == 3


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
    }
