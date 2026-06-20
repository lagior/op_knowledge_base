"""Status and observability for the knowledge base."""

from collections import defaultdict

from op_knowledge_base.change_detection import load_state
from op_knowledge_base.store import get_all_metadata, init_store


def get_status(config: dict) -> dict:
    """Gather status info: doc counts, chunk counts, last ingestion, stale docs."""
    store = init_store(config)
    all_meta = get_all_metadata(store)

    # Count docs and chunks by source type
    source_docs = defaultdict(set)
    source_chunks = defaultdict(int)
    for meta in all_meta:
        source_type = meta.get("source_type", "unknown")
        doc_id = meta.get("doc_id", "unknown")
        source_docs[source_type].add(doc_id)
        source_chunks[source_type] += 1

    # Load state files for last ingestion timestamps
    sources_info = {}
    for source_type in ["confluence", "git"]:
        state = load_state(source_type)
        doc_count = len(source_docs.get(source_type, set()))
        chunk_count = source_chunks.get(source_type, 0)

        # Find most recent last_updated across all docs in state
        last_ingested = None
        if state:
            last_ingested = max(s.last_updated for s in state.values())

        # Stale docs: in state but not in store
        state_doc_ids = set(state.keys())
        store_doc_ids = source_docs.get(source_type, set())
        stale_docs = state_doc_ids - store_doc_ids
        orphaned_docs = store_doc_ids - state_doc_ids

        sources_info[source_type] = {
            "documents": doc_count,
            "chunks": chunk_count,
            "last_ingested": last_ingested,
            "stale_docs": sorted(stale_docs),
            "orphaned_docs": sorted(orphaned_docs),
        }

    return {
        "total_chunks": len(all_meta),
        "sources": sources_info,
    }
