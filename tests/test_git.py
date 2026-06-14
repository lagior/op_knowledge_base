"""Tests for the Git source with change detection."""

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from op_knowledge_base.sources.git import fetch_changed_documents


FAKE_CONFIG = {
    "git": {
        "repos": [
            {"path": "/fake/repo", "extensions": [".md", ".py"]},
        ],
    },
}


def _make_doc(file_path: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={"file_path": file_path},
    )


@patch("op_knowledge_base.sources.git.save_state")
@patch("op_knowledge_base.sources.git.load_state")
@patch("op_knowledge_base.sources.git._build_loader")
def test_all_new_files_detected(mock_build_loader, mock_load_state, mock_save_state):
    """First run: all files are new."""
    mock_load_state.return_value = {}
    loader_instance = MagicMock()
    loader_instance.load.return_value = [
        _make_doc("README.md", "Hello world"),
        _make_doc("main.py", "print('hi')"),
    ]
    mock_build_loader.return_value = loader_instance

    changed, deleted = fetch_changed_documents(config=FAKE_CONFIG)

    assert len(changed) == 2
    assert len(deleted) == 0
    assert changed[0].metadata["doc_id"] == "git:repo:README.md"
    assert changed[1].metadata["doc_id"] == "git:repo:main.py"


@patch("op_knowledge_base.sources.git.save_state")
@patch("op_knowledge_base.sources.git.load_state")
@patch("op_knowledge_base.sources.git._build_loader")
def test_unchanged_files_skipped(mock_build_loader, mock_load_state, mock_save_state):
    """Second run with no changes: nothing returned."""
    from op_knowledge_base.change_detection import content_hash
    from op_knowledge_base.models import SourceState

    mock_load_state.return_value = {
        "git:repo:README.md": SourceState(
            doc_id="git:repo:README.md",
            source_type="git",
            content_hash=content_hash("Hello world"),
            last_updated="2024-01-01T00:00:00",
            title="README.md",
        ),
    }
    loader_instance = MagicMock()
    loader_instance.load.return_value = [
        _make_doc("README.md", "Hello world"),
    ]
    mock_build_loader.return_value = loader_instance

    changed, deleted = fetch_changed_documents(config=FAKE_CONFIG)

    assert len(changed) == 0
    assert len(deleted) == 0


@patch("op_knowledge_base.sources.git.save_state")
@patch("op_knowledge_base.sources.git.load_state")
@patch("op_knowledge_base.sources.git._build_loader")
def test_modified_file_detected(mock_build_loader, mock_load_state, mock_save_state):
    """A file with changed content is returned."""
    from op_knowledge_base.change_detection import content_hash
    from op_knowledge_base.models import SourceState

    mock_load_state.return_value = {
        "git:repo:README.md": SourceState(
            doc_id="git:repo:README.md",
            source_type="git",
            content_hash=content_hash("Old content"),
            last_updated="2024-01-01T00:00:00",
            title="README.md",
        ),
    }
    loader_instance = MagicMock()
    loader_instance.load.return_value = [
        _make_doc("README.md", "New content"),
    ]
    mock_build_loader.return_value = loader_instance

    changed, deleted = fetch_changed_documents(config=FAKE_CONFIG)

    assert len(changed) == 1
    assert changed[0].page_content == "New content"


@patch("op_knowledge_base.sources.git.save_state")
@patch("op_knowledge_base.sources.git.load_state")
@patch("op_knowledge_base.sources.git._build_loader")
def test_deleted_file_detected(mock_build_loader, mock_load_state, mock_save_state):
    """A file removed from the repo is reported as deleted."""
    from op_knowledge_base.change_detection import content_hash
    from op_knowledge_base.models import SourceState

    mock_load_state.return_value = {
        "git:repo:README.md": SourceState(
            doc_id="git:repo:README.md",
            source_type="git",
            content_hash=content_hash("Hello"),
            last_updated="2024-01-01T00:00:00",
            title="README.md",
        ),
        "git:repo:old.py": SourceState(
            doc_id="git:repo:old.py",
            source_type="git",
            content_hash=content_hash("gone"),
            last_updated="2024-01-01T00:00:00",
            title="old.py",
        ),
    }
    loader_instance = MagicMock()
    # Only README.md remains
    loader_instance.load.return_value = [
        _make_doc("README.md", "Hello"),
    ]
    mock_build_loader.return_value = loader_instance

    changed, deleted = fetch_changed_documents(config=FAKE_CONFIG)

    assert len(changed) == 0
    assert deleted == ["git:repo:old.py"]
