"""Tests for the Confluence source with change detection."""

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from op_knowledge_base.sources.confluence import fetch_changed_documents


FAKE_CONFIG = {
    "confluence": {
        "url": "https://test.atlassian.net",
        "username": "user@example.com",
        "api_token": "fake-token",
        "space_key": "TEST",
    }
}


def _make_doc(page_id: str, title: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={"id": page_id, "title": title},
    )


@patch("op_knowledge_base.sources.confluence.load_state")
@patch("op_knowledge_base.sources.confluence._build_loader")
def test_all_new_documents_detected(mock_loader, mock_load_state):
    """First run: all documents are new."""
    mock_load_state.return_value = {}
    loader_instance = MagicMock()
    loader_instance.load.return_value = [
        _make_doc("1", "Page One", "Hello world"),
        _make_doc("2", "Page Two", "Goodbye world"),
    ]
    mock_loader.return_value = loader_instance

    changed, deleted, _state = fetch_changed_documents(config=FAKE_CONFIG)

    assert len(changed) == 2
    assert len(deleted) == 0
    assert changed[0].metadata["doc_id"] == "confluence:1"
    assert changed[1].metadata["doc_id"] == "confluence:2"


@patch("op_knowledge_base.sources.confluence.load_state")
@patch("op_knowledge_base.sources.confluence._build_loader")
def test_unchanged_documents_skipped(mock_loader, mock_load_state):
    """Second run with no changes: nothing returned."""
    from op_knowledge_base.change_detection import content_hash
    from op_knowledge_base.models import SourceState

    mock_load_state.return_value = {
        "confluence:1": SourceState(
            doc_id="confluence:1",
            source_type="confluence",
            content_hash=content_hash("Hello world"),
            last_updated="2024-01-01T00:00:00",
            title="Page One",
        ),
    }
    loader_instance = MagicMock()
    loader_instance.load.return_value = [
        _make_doc("1", "Page One", "Hello world"),
    ]
    mock_loader.return_value = loader_instance

    changed, deleted, _state = fetch_changed_documents(config=FAKE_CONFIG)

    assert len(changed) == 0
    assert len(deleted) == 0


@patch("op_knowledge_base.sources.confluence.load_state")
@patch("op_knowledge_base.sources.confluence._build_loader")
def test_modified_document_detected(mock_loader, mock_load_state):
    """A document with changed content is returned."""
    from op_knowledge_base.change_detection import content_hash
    from op_knowledge_base.models import SourceState

    mock_load_state.return_value = {
        "confluence:1": SourceState(
            doc_id="confluence:1",
            source_type="confluence",
            content_hash=content_hash("Old content"),
            last_updated="2024-01-01T00:00:00",
            title="Page One",
        ),
    }
    loader_instance = MagicMock()
    loader_instance.load.return_value = [
        _make_doc("1", "Page One", "New content"),
    ]
    mock_loader.return_value = loader_instance

    changed, deleted, _state = fetch_changed_documents(config=FAKE_CONFIG)

    assert len(changed) == 1
    assert changed[0].page_content == "New content"


@patch("op_knowledge_base.sources.confluence.load_state")
@patch("op_knowledge_base.sources.confluence._build_loader")
def test_deleted_document_detected(mock_loader, mock_load_state):
    """A page removed from Confluence is reported as deleted."""
    from op_knowledge_base.change_detection import content_hash
    from op_knowledge_base.models import SourceState

    mock_load_state.return_value = {
        "confluence:1": SourceState(
            doc_id="confluence:1",
            source_type="confluence",
            content_hash=content_hash("Hello"),
            last_updated="2024-01-01T00:00:00",
            title="Page One",
        ),
        "confluence:2": SourceState(
            doc_id="confluence:2",
            source_type="confluence",
            content_hash=content_hash("World"),
            last_updated="2024-01-01T00:00:00",
            title="Page Two",
        ),
    }
    loader_instance = MagicMock()
    # Only page 1 remains
    loader_instance.load.return_value = [
        _make_doc("1", "Page One", "Hello"),
    ]
    mock_loader.return_value = loader_instance

    changed, deleted, _state = fetch_changed_documents(config=FAKE_CONFIG)

    assert len(changed) == 0
    assert deleted == ["confluence:2"]
