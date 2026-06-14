"""Tests for the query pipeline."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from op_knowledge_base.query import ask, _format_context, _build_sources


def _make_result(doc_id, content="chunk text", source_type="confluence", title="Page", score=0.8):
    """Create a (Document, score) tuple as returned by similarity search."""
    doc = Document(
        page_content=content,
        metadata={"doc_id": doc_id, "source_type": source_type, "title": title},
    )
    return (doc, score)


def _fake_config():
    return {
        "openai_api_key": "test-key",
        "embedding": {"model": "text-embedding-3-small"},
        "generation": {"model": "gpt-4o-mini"},
        "chroma": {
            "persist_directory": "/tmp/test_chroma",
            "collection_name": "test",
        },
    }


def test_format_context_single():
    """Formats a single result into labeled context."""
    results = [_make_result("c:1", "hello world", "confluence", "My Page")]
    ctx = _format_context(results)
    assert "[confluence: My Page]" in ctx
    assert "hello world" in ctx


def test_format_context_multiple():
    """Multiple results are separated by dividers."""
    results = [
        _make_result("c:1", "first"),
        _make_result("c:2", "second"),
    ]
    ctx = _format_context(results)
    assert "---" in ctx
    assert "first" in ctx
    assert "second" in ctx


def test_build_sources_deduplicates():
    """Sources are deduplicated by doc_id."""
    results = [
        _make_result("c:1", "chunk 1"),
        _make_result("c:1", "chunk 2"),
        _make_result("c:2", "chunk 3"),
    ]
    sources = _build_sources(results)
    assert len(sources) == 2
    doc_ids = {s["doc_id"] for s in sources}
    assert doc_ids == {"c:1", "c:2"}


@patch("op_knowledge_base.query.ChatOpenAI")
@patch("op_knowledge_base.query.vector_query")
@patch("op_knowledge_base.query.init_store")
def test_ask_returns_answer_and_sources(mock_init_store, mock_query, mock_llm_class):
    """Full pipeline returns answer and deduplicated sources."""
    mock_init_store.return_value = MagicMock()
    mock_query.return_value = [
        _make_result("c:1", "relevant text", "confluence", "Page A"),
        _make_result("g:1", "more text", "git", "readme.md"),
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="The answer is 42.")
    mock_llm_class.return_value = mock_llm

    result = ask(_fake_config(), "What is the answer?")

    assert result["answer"] == "The answer is 42."
    assert len(result["sources"]) == 2
    mock_llm.invoke.assert_called_once()


@patch("op_knowledge_base.query.vector_query")
@patch("op_knowledge_base.query.init_store")
def test_ask_no_results(mock_init_store, mock_query):
    """Returns a message when no documents are found."""
    mock_init_store.return_value = MagicMock()
    mock_query.return_value = []

    result = ask(_fake_config(), "Unknown topic?")

    assert "No relevant documents" in result["answer"]
    assert result["sources"] == []
