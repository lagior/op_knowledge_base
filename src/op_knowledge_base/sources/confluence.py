"""Confluence source with change detection."""

from datetime import datetime

from langchain_community.document_loaders import ConfluenceLoader
from langchain_core.documents import Document

from op_knowledge_base.change_detection import (
    content_hash,
    has_changed,
    load_state,
    save_state,
)
from op_knowledge_base.config import load_config
from op_knowledge_base.models import SourceState


SOURCE_TYPE = "confluence"


def _build_loader(config: dict) -> ConfluenceLoader:
    """Create a ConfluenceLoader from config."""
    conf = config["confluence"]
    return ConfluenceLoader(
        url=conf["url"],
        username=conf["username"],
        api_key=conf["api_token"],
    )


def _doc_id(page_id: str) -> str:
    """Build a stable document ID for a Confluence page."""
    return f"confluence:{page_id}"


def fetch_changed_documents(config: dict | None = None) -> tuple[list[Document], list[str]]:
    """Fetch Confluence pages and return only those that changed.

    Returns:
        A tuple of (changed_documents, deleted_doc_ids).
        changed_documents: Documents whose content has changed since last run.
        deleted_doc_ids: Doc IDs that were previously indexed but no longer exist.
    """
    if config is None:
        config = load_config()

    conf = config["confluence"]
    loader = _build_loader(config)

    # Load all pages from the configured space
    all_docs = loader.load(space_key=conf["space_key"], limit=500)

    state = load_state(SOURCE_TYPE)
    changed_docs = []
    seen_ids = set()

    for doc in all_docs:
        page_id = doc.metadata.get("id", doc.metadata.get("page_id", ""))
        doc_id = _doc_id(str(page_id))
        seen_ids.add(doc_id)

        new_hash = content_hash(doc.page_content)

        if has_changed(doc_id, new_hash, state):
            # Tag metadata for downstream use
            doc.metadata["doc_id"] = doc_id
            doc.metadata["source_type"] = SOURCE_TYPE
            changed_docs.append(doc)

            # Update state
            state[doc_id] = SourceState(
                doc_id=doc_id,
                source_type=SOURCE_TYPE,
                content_hash=new_hash,
                last_updated=datetime.now().isoformat(),
                title=doc.metadata.get("title", ""),
            )

    # Detect deletions: pages in state but not in current fetch
    deleted_ids = [did for did in state if did not in seen_ids]
    for did in deleted_ids:
        del state[did]

    save_state(SOURCE_TYPE, state)
    return changed_docs, deleted_ids
