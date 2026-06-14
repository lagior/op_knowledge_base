"""Tests for the vector store wrapper."""

import shutil
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from op_knowledge_base.store import (
    add_documents,
    delete_by_doc_id,
    list_doc_ids,
    query,
)


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for testing (no API calls)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        # Simple hash-based embedding: 10 dimensions
        h = hash(text)
        return [float((h >> i) & 0xFF) / 255.0 for i in range(10)]


@pytest.fixture
def store(tmp_path):
    """Create a temporary Chroma store with fake embeddings."""
    s = Chroma(
        collection_name="test",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    yield s


@pytest.fixture
def sample_docs():
    """Two documents with different doc_ids."""
    return [
        Document(
            page_content="Python is a programming language.",
            metadata={"doc_id": "doc1", "source_type": "git", "title": "Python"},
        ),
        Document(
            page_content="ChromaDB is a vector database.",
            metadata={"doc_id": "doc2", "source_type": "confluence", "title": "Chroma"},
        ),
    ]


def test_add_and_query(store, sample_docs):
    add_documents(store, sample_docs)
    results = query(store, "programming language")
    assert len(results) > 0
    # Each result is a (Document, score) tuple
    doc, score = results[0]
    assert doc.page_content in [d.page_content for d in sample_docs]


def test_add_returns_ids(store, sample_docs):
    ids = add_documents(store, sample_docs)
    assert len(ids) == 2


def test_list_doc_ids(store, sample_docs):
    add_documents(store, sample_docs)
    ids = list_doc_ids(store)
    assert ids == {"doc1", "doc2"}


def test_delete_by_doc_id(store, sample_docs):
    add_documents(store, sample_docs)
    delete_by_doc_id(store, "doc1")
    ids = list_doc_ids(store)
    assert ids == {"doc2"}


def test_delete_nonexistent_doc_id(store):
    # Should not raise
    delete_by_doc_id(store, "nonexistent")


def test_query_with_filter(store, sample_docs):
    add_documents(store, sample_docs)
    results = query(store, "database", filters={"source_type": "confluence"})
    assert len(results) > 0
    for doc, _ in results:
        assert doc.metadata["source_type"] == "confluence"


def test_list_doc_ids_empty(store):
    ids = list_doc_ids(store)
    assert ids == set()
