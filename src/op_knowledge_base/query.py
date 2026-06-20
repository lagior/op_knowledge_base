"""Query pipeline: retrieve context from vector store, generate answer with OpenAI."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from op_knowledge_base.store import init_store, query as vector_query

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context. "
    "Base your answer on the context below. If the context doesn't contain enough "
    "information to answer fully, say so.\n\n"
    "IMPORTANT: Always cite your sources by referencing the document title and type "
    "shown in square brackets (e.g. [confluence: Page Title]). When combining "
    "information from multiple sources, cite each one."
)

CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}"""


def _format_context(results: list) -> str:
    """Format retrieved documents into a context string."""
    parts = []
    for doc, score in results:
        meta = doc.metadata
        source_label = meta.get("title", meta.get("doc_id", "unknown"))
        source_type = meta.get("source_type", "unknown")
        last_updated = meta.get("last_updated", "")
        header = f"[{source_type}: {source_label}]"
        if last_updated:
            header += f" (updated: {last_updated})"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_sources(results: list) -> list[dict]:
    """Extract source references from retrieved documents."""
    seen = set()
    sources = []
    for doc, score in results:
        meta = doc.metadata
        doc_id = meta.get("doc_id", "unknown")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        sources.append({
            "doc_id": doc_id,
            "title": meta.get("title", doc_id),
            "source_type": meta.get("source_type", "unknown"),
            "last_updated": meta.get("last_updated", ""),
            "score": score,
        })
    return sources


def ask(config: dict, question: str, top_k: int = 5, source_type: str | None = None) -> dict:
    """Run the full query pipeline: retrieve -> format -> generate.

    Returns dict with 'answer' and 'sources' keys.
    source_type filters results to a specific source (e.g. 'confluence', 'git').
    """
    store = init_store(config)
    filters = {"source_type": source_type} if source_type else None
    results = vector_query(store, question, top_k=top_k, filters=filters)

    if not results:
        return {"answer": "No relevant documents found.", "sources": []}

    context = _format_context(results)
    sources = _build_sources(results)

    llm = ChatOpenAI(
        model=config["generation"]["model"],
        api_key=config["openai_api_key"],
        model_kwargs={"store": False},
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=CONTEXT_TEMPLATE.format(context=context, question=question)),
    ]

    try:
        response = llm.invoke(messages)
    except Exception as e:
        logger.error("OpenAI generation failed: %s", e)
        raise RuntimeError(f"Failed to generate answer: {e}") from e

    return {"answer": response.content, "sources": sources}
