"""Vector store wrapper over LangChain's Chroma integration."""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def init_store(config: dict) -> Chroma:
    """Initialize a persistent ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(
        model=config["embedding"]["model"],
        api_key=config["openai_api_key"],
    )
    return Chroma(
        collection_name=config["chroma"]["collection_name"],
        embedding_function=embeddings,
        persist_directory=config["chroma"]["persist_directory"],
    )


def add_documents(store: Chroma, docs: list) -> list[str]:
    """Add LangChain Document objects to the store. Returns their IDs."""
    return store.add_documents(docs)


def query(store: Chroma, question: str, top_k: int = 5, filters: dict | None = None) -> list:
    """Similarity search. Returns list of (Document, score) tuples."""
    return store.similarity_search_with_score(
        query=question,
        k=top_k,
        filter=filters,
    )


def get_chunk_hashes(store: Chroma, doc_id: str) -> dict[str, list[str]]:
    """Return {chunk_hash: [chroma_ids]} for all chunks of a document."""
    results = store.get(where={"doc_id": doc_id}, include=["metadatas"])
    hash_to_ids: dict[str, list[str]] = {}
    for chroma_id, meta in zip(results["ids"], results["metadatas"]):
        chunk_hash = meta.get("chunk_hash", "")
        hash_to_ids.setdefault(chunk_hash, []).append(chroma_id)
    return hash_to_ids


def delete_by_ids(store: Chroma, ids: list[str]) -> None:
    """Remove specific chunks by their ChromaDB IDs."""
    if ids:
        store.delete(ids=ids)


def delete_by_doc_id(store: Chroma, doc_id: str) -> None:
    """Remove all chunks belonging to a document."""
    results = store.get(where={"doc_id": doc_id})
    if results["ids"]:
        store.delete(ids=results["ids"])


def list_doc_ids(store: Chroma) -> set[str]:
    """Return all unique doc_id values in the store."""
    results = store.get(include=["metadatas"])
    return {m["doc_id"] for m in results["metadatas"] if "doc_id" in m}


def get_all_metadata(store: Chroma) -> list[dict]:
    """Return metadata for all chunks in the store."""
    results = store.get(include=["metadatas"])
    return results["metadatas"]
