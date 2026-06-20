"""Git repository source with change detection."""

from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document

from op_knowledge_base.change_detection import (
    content_hash,
    has_changed,
    load_state,
)
from op_knowledge_base.config import load_config
from op_knowledge_base.models import SourceState


SOURCE_TYPE = "git"


def _doc_id(repo_name: str, file_path: str) -> str:
    """Build a stable document ID for a git file."""
    return f"git:{repo_name}:{file_path}"


def _build_loader(repo_config: dict) -> GitLoader:
    """Create a GitLoader for a single repo config entry."""
    extensions = set(repo_config.get("extensions", [".md", ".py", ".txt", ".yaml", ".yml"]))

    def file_filter(path: str) -> bool:
        return Path(path).suffix in extensions

    branch = repo_config.get("branch", "main")
    return GitLoader(
        repo_path=repo_config["path"],
        branch=branch,
        file_filter=file_filter,
    )


def fetch_changed_documents(config: dict | None = None) -> tuple[list[Document], list[str], dict]:
    """Fetch files from git repos and return only those that changed.

    Returns:
        A tuple of (changed_documents, deleted_doc_ids, new_state).
        State is returned but NOT saved; the caller saves after successful storage.
    """
    if config is None:
        config = load_config()

    state = load_state(SOURCE_TYPE)
    changed_docs = []
    seen_ids = set()

    for repo_config in config["git"]["repos"]:
        repo_name = Path(repo_config["path"]).name
        loader = _build_loader(repo_config)
        all_docs = loader.load()

        for doc in all_docs:
            file_path = doc.metadata.get("file_path", "")
            doc_id = _doc_id(repo_name, file_path)
            seen_ids.add(doc_id)

            new_hash = content_hash(doc.page_content)

            if has_changed(doc_id, new_hash, state):
                doc.metadata["doc_id"] = doc_id
                doc.metadata["source_type"] = SOURCE_TYPE
                doc.metadata["repo_name"] = repo_name
                doc.metadata["file_path"] = file_path
                changed_docs.append(doc)

                state[doc_id] = SourceState(
                    doc_id=doc_id,
                    source_type=SOURCE_TYPE,
                    content_hash=new_hash,
                    last_updated=datetime.now().isoformat(),
                    title=file_path,
                )

    # Detect deletions: files in state but not in current fetch
    deleted_ids = [did for did in state if did not in seen_ids]
    for did in deleted_ids:
        del state[did]

    return changed_docs, deleted_ids, state
