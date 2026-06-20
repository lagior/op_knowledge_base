"""Tests for the status module."""

import json

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from unittest.mock import patch

from op_knowledge_base.status import get_status
from op_knowledge_base.store import add_documents


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        h = hash(text)
        return [float((h >> i) & 0xFF) / 255.0 for i in range(10)]


@pytest.fixture
def fake_store(tmp_path):
    return Chroma(
        collection_name="test",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )


@pytest.fixture
def config(tmp_path):
    return {
        "openai_api_key": "fake",
        "embedding": {"model": "text-embedding-3-small"},
        "chroma": {
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": "test",
        },
    }


def _write_state(tmp_path, source_type, state_data):
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    with open(state_dir / f"{source_type}.json", "w") as f:
        json.dump(state_data, f)


def test_status_empty(config, tmp_path):
    """Status on empty store returns zeros."""
    with patch("op_knowledge_base.status.init_store") as mock_init, \
         patch("op_knowledge_base.status.load_state", return_value={}):
        store = Chroma(
            collection_name="test",
            embedding_function=FakeEmbeddings(),
            persist_directory=str(tmp_path / "chroma"),
        )
        mock_init.return_value = store
        info = get_status(config)

    assert info["total_chunks"] == 0
    assert info["sources"]["confluence"]["documents"] == 0
    assert info["sources"]["git"]["documents"] == 0


def test_status_with_docs(config, fake_store, tmp_path):
    """Status counts docs and chunks by source type."""
    docs = [
        Document(page_content="chunk 1", metadata={"doc_id": "d1", "source_type": "git"}),
        Document(page_content="chunk 2", metadata={"doc_id": "d1", "source_type": "git"}),
        Document(page_content="chunk 3", metadata={"doc_id": "d2", "source_type": "confluence"}),
    ]
    add_documents(fake_store, docs)

    with patch("op_knowledge_base.status.init_store", return_value=fake_store), \
         patch("op_knowledge_base.status.load_state", return_value={}):
        info = get_status(config)

    assert info["total_chunks"] == 3
    assert info["sources"]["git"]["documents"] == 1
    assert info["sources"]["git"]["chunks"] == 2
    assert info["sources"]["confluence"]["documents"] == 1
    assert info["sources"]["confluence"]["chunks"] == 1


def test_status_last_ingested(config, fake_store):
    """Status shows most recent last_updated from state."""
    from op_knowledge_base.models import SourceState

    state = {
        "d1": SourceState("d1", "git", "abc", "2024-01-01T00:00:00"),
        "d2": SourceState("d2", "git", "def", "2024-06-15T12:00:00"),
    }

    def mock_load(source_type):
        return state if source_type == "git" else {}

    with patch("op_knowledge_base.status.init_store", return_value=fake_store), \
         patch("op_knowledge_base.status.load_state", side_effect=mock_load):
        info = get_status(config)

    assert info["sources"]["git"]["last_ingested"] == "2024-06-15T12:00:00"
    assert info["sources"]["confluence"]["last_ingested"] is None


def test_status_stale_docs(config, fake_store):
    """Docs in state but not in store are reported as stale."""
    from op_knowledge_base.models import SourceState

    state = {
        "d1": SourceState("d1", "git", "abc", "2024-01-01T00:00:00"),
        "d_missing": SourceState("d_missing", "git", "xyz", "2024-01-01T00:00:00"),
    }
    # Only d1 is in the store
    docs = [Document(page_content="chunk", metadata={"doc_id": "d1", "source_type": "git"})]
    add_documents(fake_store, docs)

    def mock_load(source_type):
        return state if source_type == "git" else {}

    with patch("op_knowledge_base.status.init_store", return_value=fake_store), \
         patch("op_knowledge_base.status.load_state", side_effect=mock_load):
        info = get_status(config)

    assert "d_missing" in info["sources"]["git"]["stale_docs"]
    assert "d1" not in info["sources"]["git"]["stale_docs"]
